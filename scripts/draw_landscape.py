"""Build the original vector mountain backdrop. No photos or external assets.

Fixed ridge anchors establish the composition; seeded subdivisions and branching
gullies give the slopes detail without a repeating triangular mountain motif.
Run from any directory to regenerate web/mountains.svg.
"""
from pathlib import Path
import math
import random

rng = random.Random(1729)
W, H = 1600, 400
parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 400" fill="none">',
         '<defs><linearGradient id="haze" x2="0" y2="1"><stop stop-color="#edf4f4" stop-opacity="0"/><stop offset="1" stop-color="#edf4f4" stop-opacity=".82"/></linearGradient>',
         '<linearGradient id="water" x2="0" y2="1"><stop stop-color="#c8dcde"/><stop offset="1" stop-color="#f6f5ef"/></linearGradient></defs>']


def path(points):
    return 'M' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)


def ridge(anchors, roughness):
    points = [anchors[0]]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        count = max(2, round((x1 - x0) / 5))
        for i in range(1, count + 1):
            t = i / count
            y = y0 + (y1 - y0) * t
            y += rng.uniform(-roughness, roughness) * math.sin(math.pi * t)
            points.append((x0 + (x1 - x0) * t, y))
    return points


def mountain(name, anchors, base, shade, snow, roughness, floor, detail):
    points = ridge(anchors, roughness)
    outline = path(points) + f' L1600,{floor} L0,{floor}Z'
    parts.append(f'<defs><clipPath id="{name}"><path d="{outline}"/></clipPath></defs>')
    parts.append(f'<path d="{outline}" fill="{base}"/>')
    parts.append(f'<g clip-path="url(#{name})">')
    if name in ('distant', 'middle'):
        # Snow follows the actual crest, with an uneven lower snowline and
        # deeper coverage at higher elevations. Rock faces below shade the cap.
        snow_edge = []
        for x, y in points:
            if name == 'distant':
                coverage = 14 + max(0, 115 - y) * .32
                coverage += 5 * math.sin(x / 13) + 3 * math.sin(x / 5)
            else:
                coverage = max(0, 116 - y) * .45
                coverage += min(coverage, 3) * math.sin(x / 9)
            snow_edge.append((x, y + max(0, coverage)))
        parts.append(f'<path d="{path(points + list(reversed(snow_edge)))}Z" fill="{snow}"/>')
    # Broad overlapping faces descend from individual summits. Their tributaries
    # converge into valleys rather than running in parallel down every slope.
    for j in range(1, len(anchors) - 1):
        x, y = anchors[j]
        if y > anchors[j - 1][1] and y > anchors[j + 1][1]:
            continue
        depth = (floor - y) * rng.uniform(.7, 1.15)
        drift = rng.uniform(30, 125)
        width = rng.uniform(25, 90)
        center = [(x, y - 2)]
        for k in range(1, 7):
            t = k / 6
            center.append((x + drift * t + rng.uniform(-12, 12), y + depth * t))
        edge = [(cx + width * k / 6, cy + rng.uniform(-4, 4))
                for k, (cx, cy) in reversed(list(enumerate(center)))]
        parts.append(f'<path d="{path(center + edge)}Z" fill="{shade}" opacity="{rng.uniform(.12,.35):.2f}"/>')
        if name == 'distant':
            # Uneven snow fields and exposed rock on the higher, distant range.
            for branch in range(6):
                start_x = x + rng.uniform(-22, 30)
                start_y = y + rng.uniform(-3, 18)
                end_x = x + drift * rng.uniform(.25, .8) + rng.uniform(-25, 25)
                end_y = y + depth * rng.uniform(.25, .7)
                snowline = [(start_x, start_y)]
                for step in range(1, 7):
                    t = step / 6
                    snowline.append((start_x + (end_x - start_x) * t + rng.uniform(-5, 5),
                                     start_y + (end_y - start_y) * t))
                snowedge = [(sx + rng.uniform(1, 5), sy + rng.uniform(-2, 2))
                            for sx, sy in reversed(snowline)]
                parts.append(f'<path d="{path(snowline + snowedge)}Z" fill="{snow}" opacity=".7"/>')
        # Fine, broken rock contours follow the fall of the terrain.
        for contour in range(16):
            start = rng.uniform(.1, .75)
            length = rng.uniform(.04, .16)
            offset = rng.uniform(0, width)
            trace = []
            for step in range(5):
                t = start + length * step / 4
                trace.append((x + (drift + offset) * t + rng.uniform(-3, 3), y + depth * t))
            parts.append(f'<path d="{path(trace)}" stroke="{shade}" stroke-width=".7" opacity=".13"/>')
    parts.append(f'<path d="{outline}" fill="url(#haze)"/>')
    parts.append('</g>')


mountain('distant', [(0,132),(42,109),(78,121),(131,83),(164,96),(201,72),
    (242,97),(291,70),(325,80),(363,45),(395,61),(432,27),(454,39),(478,20),
    (511,50),(548,42),(575,64),(610,53),(638,78),(674,50),(697,64),
    (729,46),(756,71),(794,83),(831,65),(868,88),(902,78),(936,104),
    (984,87),(1025,56),(1050,66),(1092,32),(1115,48),(1145,39),
    (1190,76),(1226,65),(1260,91),(1300,72),(1336,94),(1380,83),
    (1416,111),(1462,85),(1500,99),(1543,77),(1600,103)],
    '#b9cdd4','#7895a6','#fcfdfb',5,260,2)

mountain('middle', [(0,198),(55,168),(95,178),(145,149),(188,155),
    (234,122),(262,135),(293,114),(326,119),(359,95),(390,105),
    (423,91),(456,113),(487,104),(518,128),(554,139),(590,129),
    (630,150),(664,145),(710,175),(746,163),(785,172),(824,150),
    (858,155),(897,126),(936,139),(970,110),(1005,117),(1040,91),
    (1072,104),(1113,84),(1155,99),(1188,93),(1230,118),(1263,112),
    (1302,138),(1342,125),(1380,144),(1420,133),(1465,156),(1520,147),(1600,178)],
    '#a3b9ad','#668476','#edf3ec',3,282,3)

mountain('foothills', [(0,233),(65,211),(103,214),(143,203),(175,209),
    (222,193),(262,198),(313,186),(355,193),(401,208),(452,222),
    (497,217),(555,232),(615,241),(680,228),(738,235),(792,219),
    (850,211),(893,194),(943,202),(994,188),(1038,195),(1090,174),
    (1134,184),(1181,171),(1225,184),(1270,194),(1308,191),
    (1356,210),(1402,201),(1450,217),(1493,213),(1544,226),(1600,217)],
    '#819f85','#4e735b','#b8cebb',1.8,290,4)

# A fine, irregular wooded shoreline and a few tall poplar silhouettes.
parts.append('<path d="M0 289 Q260 281 500 290T980 288T1600 287V400H0Z" fill="url(#water)"/>')
for i in range(450):
    x = rng.uniform(0, W)
    y = 284 + 3 * math.sin(x / 110)
    height = rng.uniform(2, 9)
    if rng.random() < .06:
        height = rng.uniform(14, 26)
        width = height * .12
        parts.append(f'<path d="M{x:.1f},{y:.1f}q{-width*2:.1f},{-height*.45:.1f} 0,{-height:.1f}q{width*2:.1f},{height*.55:.1f} 0,{height:.1f}Z" fill="#537c62" opacity=".65"/>')
    else:
        width = height * .36
        parts.append(f'<path d="M{x-width:.1f},{y:.1f}l{width:.1f},{-height:.1f} {width:.1f},{height:.1f}Z" fill="#719078" opacity=".45"/>')

for i in range(180):
    x = rng.uniform(-30, W)
    y = rng.uniform(294, 390)
    length = rng.uniform(5, 65) * (y - 270) / 60
    parts.append(f'<path d="M{x:.1f},{y:.1f}h{length:.1f}" stroke="{rng.choice(["#ffffff", "#96b7bd"])}" stroke-width="{rng.uniform(.4,1):.1f}" opacity="{(400-y)/450:.2f}"/>')
parts.append('</svg>')
target = Path(__file__).resolve().parents[1] / 'web' / 'mountains.svg'
target.write_text('\n'.join(parts) + '\n', encoding='utf-8')
print(f'Wrote {target.name}: {target.stat().st_size:,} bytes')
