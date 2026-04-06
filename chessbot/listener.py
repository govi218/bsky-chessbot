"""Bluesky listener for chess bot mentions."""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import httpx
from PIL import Image
import io


@dataclass
class Mention:
    """A mention of the bot in a Bluesky post."""
    did: str  # Author's DID
    handle: str  # Author's handle
    text: str  # Post text
    uri: str  # AT URI of the post
    cid: str  # Content ID
    images: list[str]  # Image URLs
    timestamp: datetime


class ChessBotListener:
    """Listen for @mentions on Bluesky and process chess position images."""

    def __init__(
        self,
        bot_handle: str = "chess.glados.computer",
        jetstream_url: str = "wss://jetstream2.us-east.bsky.network/subscribe"
    ):
        self.bot_handle = bot_handle
        self.jetstream_url = jetstream_url
        self.on_mention: Optional[Callable] = None
        self._cursor = None

    def _extract_images(self, event: dict) -> list[str]:
        """Extract image URLs from a post."""
        images = []
        embed = event.get("commit", {}).get("record", {}).get("embed", {})

        if embed.get("$type") == "app.bsky.embed.images":
            for image in embed.get("images", []):
                # Image ref contains blob CID
                if "image" in image:
                    # Build URL from blob ref
                    # https://bsky.social/xrpc/com.atproto.sync.getBlob?did=xxx&cid=xxx
                    blob_cid = image["image"].get("$ref") or image["image"].get("cid")
                    if blob_cid:
                        did = event["did"]
                        images.append(f"https://bsky.social/xrpc/com.atproto.sync.getBlob?did={did}&cid={blob_cid}")

        return images

    async def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download an image from URL."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=30)
                if resp.status_code == 200:
                    return Image.open(io.BytesIO(resp.content))
        except Exception as e:
            print(f"Error downloading image: {e}")
        return None

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

        # Extract images
        images = self._extract_images(event)

        # Build mention object
        mention = Mention(
            did=event["did"],
            handle=event.get("handle", "unknown"),
            text=text,
            uri=f"at://{event['did']}/app.bsky.feed.post/{commit['rkey']}",
            cid=commit.get("cid", ""),
            images=images,
            timestamp=datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
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


async def process_mention(mention: Mention):
    """Process a mention - download image and analyze."""
    from .bot import analyze, format_result

    if not mention.images:
        print(f"No images in mention from @{mention.handle}")
        return

    # Download first image
    print(f"Downloading image from @{mention.handle}...")
    listener = ChessBotListener()
    img = await listener._download_image(mention.images[0])

    if img is None:
        print("Failed to download image")
        return

    # Save temporarily
    temp_path = Path(f"/tmp/chess_{mention.cid}.png")
    img.save(temp_path)

    # Analyze
    print("Analyzing position...")
    result = analyze(temp_path)
    print(format_result(result))

    # Cleanup
    temp_path.unlink(missing_ok=True)

    return result


if __name__ == "__main__":
    listener = ChessBotListener()
    listener.on_mention = process_mention
    asyncio.run(listener.start())
