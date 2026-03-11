#  __init__.py
#
#  Copyright (c) 2025 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from itertools import islice
from typing import AsyncIterator, Tuple
from typing import Final
from urllib.parse import urlparse, parse_qs

import humanize
import requests
from bs4 import BeautifulSoup
from mcp import ServerSession
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field, BaseModel, AwareDatetime
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscriptSnippet
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig, ProxyConfig


@dataclass(frozen=True)
class AppContext:
    http_client: requests.Session
    ytt_api: YouTubeTranscriptApi


@asynccontextmanager
async def _app_lifespan(_server: FastMCP, proxy_config: ProxyConfig | None) -> AsyncIterator[AppContext]:
    with requests.Session() as http_client:
        ytt_api = YouTubeTranscriptApi(http_client=http_client, proxy_config=proxy_config)
        yield AppContext(http_client=http_client, ytt_api=ytt_api)


class Transcript(BaseModel):
    """Transcript of a YouTube video."""

    title: str = Field(description="Title of the video")
    transcript: str = Field(description="Transcript of the video")
    next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None)


class TranscriptSnippet(BaseModel):
    """Transcript snippet of a YouTube video."""

    text: str = Field(description="Text of the transcript snippet")
    start: float = Field(description="The timestamp at which this transcript snippet appears on screen in seconds.")
    duration: float = Field(description="The duration of how long the snippet in seconds.")

    def __len__(self) -> int:
        return len(self.model_dump_json())

    @classmethod
    def from_fetched_transcript_snippet(
        cls: type[TranscriptSnippet], snippet: FetchedTranscriptSnippet
    ) -> TranscriptSnippet:
        return cls(text=snippet.text, start=snippet.start, duration=snippet.duration)


class TimedTranscript(BaseModel):
    """Transcript of a YouTube video with timestamps."""

    title: str = Field(description="Title of the video")
    snippets: list[TranscriptSnippet] = Field(description="Transcript snippets of the video")
    next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None)


class VideoInfo(BaseModel):
    """Video information."""

    title: str = Field(description="Title of the video")
    description: str = Field(description="Description of the video")
    uploader: str = Field(description="Uploader of the video")
    upload_date: AwareDatetime = Field(description="Upload date of the video")
    duration: str = Field(description="Duration of the video")


def _extract_yt_initial_player_response(html: str) -> dict:
    """Extract ytInitialPlayerResponse JSON from YouTube page HTML."""
    pattern = r'var\s+ytInitialPlayerResponse\s*=\s*({.*?});\s*(?:var|</script)'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Fallback pattern
    pattern2 = r'ytInitialPlayerResponse\s*=\s*({.*?});'
    match2 = re.search(pattern2, html, re.DOTALL)
    if match2:
        return json.loads(match2.group(1))

    raise ValueError("Could not find ytInitialPlayerResponse in page HTML")


def _parse_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    else:
        q = parse_qs(parsed_url.query).get("v")
        if q is None:
            raise ValueError(f"couldn't find a video ID from the provided URL: {url}.")
        return q[0]


@lru_cache
def _get_transcript_snippets(ctx: AppContext, video_id: str, lang: str) -> Tuple[str, list[FetchedTranscriptSnippet]]:
    if lang == "en":
        languages = ["en"]
    else:
        languages = [lang, "en"]

    page = ctx.http_client.get(
        f"https://www.youtube.com/watch?v={video_id}", headers={"Accept-Language": ",".join(languages)}
    )
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    title = soup.title.string if soup.title and soup.title.string else "Transcript"

    transcripts = ctx.ytt_api.fetch(video_id, languages=languages)
    return title, transcripts.snippets


@lru_cache
def _get_video_info(ctx: AppContext, video_url: str) -> VideoInfo:
    video_id = _parse_video_id(video_url)
    page = ctx.http_client.get(
        f"https://www.youtube.com/watch?v={video_id}",
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        },
    )
    page.raise_for_status()

    player_response = _extract_yt_initial_player_response(page.text)
    video_details = player_response.get("videoDetails", {})
    microformat = player_response.get("microformat", {}).get("playerMicroformatRenderer", {})

    title = video_details.get("title", "Unknown")
    description = video_details.get("shortDescription", "")
    uploader = video_details.get("author", "Unknown")

    # Parse upload date
    date_str = microformat.get("uploadDate") or microformat.get("publishDate", "")
    if date_str:
        try:
            upload_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            upload_date = datetime.now(timezone.utc)
    else:
        upload_date = datetime.now(timezone.utc)

    # Parse duration
    duration_seconds = int(video_details.get("lengthSeconds", 0))
    duration_str = humanize.naturaldelta(timedelta(seconds=duration_seconds))

    return VideoInfo(
        title=title,
        description=description,
        uploader=uploader,
        upload_date=upload_date,
        duration=duration_str,
    )


def server(
    response_limit: int | None = None,
    webshare_proxy_username: str | None = None,
    webshare_proxy_password: str | None = None,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> FastMCP:
    """Initializes the MCP server."""

    proxy_config: ProxyConfig | None = None
    if webshare_proxy_username and webshare_proxy_password:
        proxy_config = WebshareProxyConfig(webshare_proxy_username, webshare_proxy_password)
    elif http_proxy or https_proxy:
        proxy_config = GenericProxyConfig(http_proxy, https_proxy)

    mcp = FastMCP("Youtube Transcript", lifespan=partial(_app_lifespan, proxy_config=proxy_config), host=host, port=port)

    @mcp.tool()
    async def get_transcript(
        ctx: Context[ServerSession, AppContext],
        url: str = Field(description="The URL of the YouTube video"),
        lang: str = Field(description="The preferred language for the transcript", default="en"),
        next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None),
    ) -> Transcript:
        """Retrieves the transcript of a YouTube video."""

        title, snippets = _get_transcript_snippets(ctx.request_context.lifespan_context, _parse_video_id(url), lang)
        transcripts = (item.text for item in snippets)

        if response_limit is None or response_limit <= 0:
            return Transcript(title=title, transcript="\n".join(transcripts))

        res = ""
        cursor = None
        for i, line in islice(enumerate(transcripts), int(next_cursor or 0), None):
            if len(res) + len(line) + 1 > response_limit:
                cursor = str(i)
                break
            res += f"{line}\n"

        return Transcript(title=title, transcript=res[:-1], next_cursor=cursor)

    @mcp.tool()
    async def get_timed_transcript(
        ctx: Context[ServerSession, AppContext],
        url: str = Field(description="The URL of the YouTube video"),
        lang: str = Field(description="The preferred language for the transcript", default="en"),
        next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None),
    ) -> TimedTranscript:
        """Retrieves the transcript of a YouTube video with timestamps."""

        title, snippets = _get_transcript_snippets(ctx.request_context.lifespan_context, _parse_video_id(url), lang)

        if response_limit is None or response_limit <= 0:
            return TimedTranscript(
                title=title, snippets=[TranscriptSnippet.from_fetched_transcript_snippet(s) for s in snippets]
            )

        res = []
        size = len(title) + 1
        cursor = None
        for i, s in islice(enumerate(snippets), int(next_cursor or 0), None):
            snippet = TranscriptSnippet.from_fetched_transcript_snippet(s)
            if size + len(snippet) + 1 > response_limit:
                cursor = str(i)
                break
            res.append(snippet)

        return TimedTranscript(title=title, snippets=res, next_cursor=cursor)

    @mcp.tool()
    def get_video_info(
        ctx: Context[ServerSession, AppContext],
        url: str = Field(description="The URL of the YouTube video"),
    ) -> VideoInfo:
        """Retrieves the video information."""
        return _get_video_info(ctx.request_context.lifespan_context, url)

    return mcp


__all__: Final = ["server", "Transcript", "TimedTranscript", "TranscriptSnippet", "VideoInfo"]
