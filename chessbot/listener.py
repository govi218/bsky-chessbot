"""Bluesky listener for chess bot mentions."""

import asyncio
import json
import re
import os
import time
import traceback
import httpx
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from atproto import Client

from .bot import analyze, format_result

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass
class Mention:
    """A mention of the bot in a Bluesky post."""
    did: str  # Author's DID
    handle: str  # Author's handle
    text: str  # Post text
    uri: str  # AT URI of the post
    cid: str  # Content ID
    images: list[tuple[str, str]]  # List of (did, blob_cid) tuples
    timestamp: int  # Unix microseconds
    parent_uri: Optional[str] = None  # Parent post URI if this is a reply
    parent_cid: Optional[str] = None  # Parent post CID


class ChessBotListener:
    """Listen for @mentions on Bluesky and process chess position images."""

    def __init__(
        self,
        bot_handle: str = "chess.glados.computer",
        bot_did: str = "did:plc:ulqnoe35solt5jewkxqjlofu",
        jetstream_url: str = "wss://jetstream2.us-east.bsky.network/subscribe",
        password: str | None = None
    ):
        self.bot_did = bot_did
        self.bot_handle = bot_handle
        self.jetstream_url = jetstream_url
        self.on_mention: Optional[Callable] = None
        self._cursor = None

        # Rate limiting
        self._last_mention: dict[str, int] = {}  # did -> timestamp
        self._min_interval = 60  # seconds between mentions from same user

        # Bluesky client for replies
        self.client = Client(base_url="https://pds.glados.computer")
        self._logged_in = False
        self._password = password or os.environ.get("BLUESKY_PASSWORD")

    async def login(self):
        """Login to Bluesky."""
        if self._password and not self._logged_in:
            self.client.login(self.bot_did, self._password)
            self._logged_in = True
            print(f"Logged in as @{self.bot_handle}")

    async def reply(self, mention: Mention, text: str):
        """Reply to a mention."""
        await self.login()

        # Reply to the mention post
        parent_ref = {
            "uri": mention.uri,
            "cid": mention.cid
        }

        # If mention is a reply, use its root; otherwise mention is the root
        if mention.parent_uri:
            # Need to fetch the root from the thread
            # For now, use parent as root (will be corrected if needed)
            root_ref = {"uri": mention.parent_uri, "cid": mention.parent_cid or ""}
        else:
            root_ref = parent_ref

        # Create reply
        self.client.send_post(
            text=text,
            reply_to={"parent": parent_ref, "root": root_ref}
        )

    def _check_rate_limit(self, did: str) -> bool:
        """Check if user is rate limited. Returns True if should skip."""
        now = int(time.time())
        last = self._last_mention.get(did, 0)
        if now - last < self._min_interval:
            return True
        self._last_mention[did] = now
        return False

    async def _resolve_pds(self, did: str) -> str:
        """Resolve the PDS URL for a DID."""
        # DID format: did:plc:xxx or did:web:xxx
        if did.startswith("did:plc:"):
            # PLC DIDs - resolve via PLC directory
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://plc.directory/{did}", timeout=10)
                if resp.status_code == 200:
                    doc = resp.json()
                    # Find the service endpoint for ATProto PDS
                    for service in doc.get("service", []):
                        if service.get("type") == "AtprotoPersonalDataServer":
                            return service.get("serviceEndpoint")
        elif did.startswith("did:web:"):
            # DID:web - resolve from the domain's well-known
            domain = did.replace("did:web:", "")
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://{domain}/.well-known/did.json", timeout=10)
                if resp.status_code == 200:
                    doc = resp.json()
                    for service in doc.get("service", []):
                        if service.get("type") == "AtprotoPersonalDataServer":
                            return service.get("serviceEndpoint")
        # Fallback to bsky.social
        return "https://bsky.social"

    def _extract_images(self, event: dict) -> list[tuple[str, str]]:
        """Extract image blob CIDs from a post. Returns list of (did, cid) tuples."""
        images = []
        embed = event.get("commit", {}).get("record", {}).get("embed", {})

        if embed.get("$type") == "app.bsky.embed.images":
            for image in embed.get("images", []):
                if "image" in image:
                    blob_cid = image["image"].get("ref", {}).get("$link") or image["image"].get("cid")
                    if blob_cid:
                        did = event["did"]
                        images.append((did, blob_cid))

        return images

    async def _build_blob_url(self, did: str, cid: str) -> str:
        """Build the blob URL by resolving the PDS."""
        pds = await self._resolve_pds(did)
        return f"{pds}/xrpc/com.atproto.sync.getBlob?did={did}&cid={cid}"

    async def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download an image from URL."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=30)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content))
                    # Validate size
                    if img.size[0] > 2000 or img.size[1] > 2000:
                        print("Image too large, skipping")
                        return None
                    return img
        except Exception as e:
            print(f"Error downloading image: {e}")
        return None

    async def _fetch_parent_images(self, parent_uri: str) -> list[tuple[str, str]]:
        """Fetch images from a parent post using authenticated client."""
        await self.login()

        try:
            # Use the atproto client to get the post thread
            thread = self.client.app.bsky.feed.get_post_thread({"uri": parent_uri})
            post = thread.thread.post

            # Extract images from embed
            images = []
            embed = post.record.embed if hasattr(post.record, 'embed') else None

            if embed:
                if hasattr(embed, 'images'):
                    for image in embed.images:
                        if hasattr(image, 'image') and hasattr(image.image, 'ref'):
                            blob_cid = image.image.ref.link
                            parent_did = post.author.did
                            images.append((parent_did, blob_cid))

            return images
        except Exception as e:
            print(f"Error fetching parent post: {e}")
            traceback.print_exc()
            return []

    async def _handle_event(self, event: dict):
        """Handle a Jetstream event."""
        if event.get("kind") != "commit":
            return

        commit = event.get("commit", {})
        if commit.get("collection") != "app.bsky.feed.post":
            return

        if commit.get("operation") != "create":
            return

        record = commit.get("record", {})
        text = record.get("text", "")

        # Check for mention
        if f"@{self.bot_handle}" not in text:
            return

        did = event["did"]

        # Rate limit check
        # if self._check_rate_limit(did):
            # print(f"Rate limited @{event.get('handle', did)}")
            # return

        # Content labels check
        if record.get("labels"):
            print("Content has labels, skipping")
            return

        # Extract images
        images = self._extract_images(event)

        # Extract parent info if this is a reply
        parent_uri = None
        parent_cid = None
        reply = record.get("reply", {})
        if reply:
            parent_uri = reply.get("parent", {}).get("uri")
            parent_cid = reply.get("parent", {}).get("cid")

        # Build mention object
        mention = Mention(
            did=did,
            handle=event.get("handle", "unknown"),
            text=text,
            uri=f"at://{did}/app.bsky.feed.post/{commit['rkey']}",
            cid=commit.get("cid", ""),
            images=images,
            timestamp=event["time_us"],
            parent_uri=parent_uri,
            parent_cid=parent_cid
        )

        print(f"Mention from @{mention.handle}: {text[:50]}...")

        # Call handler if set
        if self.on_mention:
            await self.on_mention(mention)

    async def start(self):
        """Start listening for mentions."""
        import websockets

        params = {"wantedCollections": "app.bsky.feed.post"}
        if self._cursor:
            params["cursor"] = self._cursor

        url = f"{self.jetstream_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

        print(f"Listening for @{self.bot_handle} mentions...")

        async with websockets.connect(url) as ws:
            async for message in ws:
                try:
                    event = json.loads(message)
                    await self._handle_event(event)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    traceback.print_exc()


async def process_mention(mention: Mention):
    """Process a mention - download image and analyze."""
    from .bot import analyze, format_result

    listener = ChessBotListener()
    images = mention.images

    # If no images in mention, check parent post
    if not images and mention.parent_uri:
        print(f"No images in mention, checking parent post... {mention}")
        parent_images = await listener._fetch_parent_images(mention.parent_uri)
        if parent_images:
            images = parent_images

    if not images:
        print(f"No images found from @{mention.handle}")
        return

    # Download first image
    print(f"Downloading image from @{mention.handle}...")
    did, cid = images[0]
    url = await listener._build_blob_url(did, cid)
    img = await listener._download_image(url)

    if img is None:
        print("Failed to download image")
        return

    # Save temporarily
    temp_path = Path(f"/tmp/chess_{mention.cid}.png")
    img.save(temp_path)

    try:
        # Analyze
        print("Analyzing position...")
        result = analyze(temp_path)

        # Check for turn preference in text
        turn = None
        text_lower = mention.text.lower()
        if "white" in text_lower:
            turn = "w"
        elif "black" in text_lower:
            turn = "b"

        reply_text = format_result(result, turn=turn)
        await listener.reply(mention, reply_text)

        return result
    finally:
        # Cleanup
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    listener = ChessBotListener()
    listener.on_mention = process_mention
    asyncio.run(listener.start())
