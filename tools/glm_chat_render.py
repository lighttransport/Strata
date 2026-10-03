"""Render an output-only asciicast clip at its original speed (requires pyte and Pillow)."""
import argparse
import json
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont


class TerminalScreen(pyte.Screen):
    # ncurses emits CSI S and CSI Z when updating long answers. pyte 0.8
    # lacks these two handlers; ignoring them leaves stale, scrambled lines.
    def scroll_up(self, count=1):
        x, y = self.cursor.x, self.cursor.y
        self.cursor.y = self.margins.bottom if self.margins else self.lines - 1
        for _ in range(count or 1):
            self.index()
        self.cursor.x, self.cursor.y = x, y

    def back_tab(self, count=1):
        for _ in range(count or 1):
            self.cursor.x = max((stop for stop in self.tabstops if stop < self.cursor.x), default=0)


class TerminalStream(pyte.Stream):
    csi = dict(pyte.Stream.csi, S='scroll_up', Z='back_tab')
    events = pyte.Stream.events | frozenset(('scroll_up', 'back_tab'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cast', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--duration', type=float, default=8)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--title', default='GLM5.3Flash Q2 - live decode / original speed')
    parser.add_argument('--font', default='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
    args = parser.parse_args()
    if args.start < 0 or not 1 <= args.fps <= 30 or round(args.duration * args.fps) < 1:
        parser.error('invalid clip timing')
    header, *events = map(json.loads, args.cast.read_text(encoding='utf-8').splitlines())
    if header.get('version') != 2 or any(event[1] != 'o' for event in events):
        parser.error('expected output-only asciicast v2')
    if not events or args.start + args.duration > events[-1][0]:
        parser.error('clip exceeds recording')
    screen = TerminalScreen(header['width'], header['height'])
    stream = TerminalStream(screen)
    font = ImageFont.truetype(args.font, 16)
    padding, line_height, title_height = 16, 22, 46
    width = int(header['width'] * font.getlength('M')) + 2 * padding
    height = title_height + header['height'] * line_height + 2 * padding
    images, index = [], 0
    for frame in range(round(args.duration * args.fps)):
        timestamp = args.start + frame / args.fps
        while index < len(events) and events[index][0] <= timestamp:
            stream.feed(events[index][2])
            index += 1
        image = Image.new('RGB', (width, height), '#0d1117')
        draw = ImageDraw.Draw(image)
        draw.text((padding, 12), args.title, font=font, fill='#76c7c0')
        for row, text in enumerate(screen.display):
            y = title_height + padding + row * line_height
            rate = text.startswith(('LIVE decode', 'FINAL decode'))
            if rate:
                draw.rectangle((10, y - 2, width - 10, y + line_height), fill='#16312a')
            draw.text((padding, y), text.rstrip(), font=font,
                      fill='#b9f0cd' if rate else '#e1e6ee')
        images.append(image)
    palette = images[0].quantize(colors=64)
    indexed = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(args.output, save_all=True, append_images=indexed[1:],
                    duration=[round((i + 1) * 1000 / args.fps) - round(i * 1000 / args.fps)
                              for i in range(len(indexed))], loop=0, optimize=True, disposal=1)
    print(json.dumps(dict(start_seconds=args.start, duration_seconds=len(images) / args.fps,
                          fps=args.fps, playback_speed=1, bytes=args.output.stat().st_size)))


if __name__ == '__main__':
    main()
