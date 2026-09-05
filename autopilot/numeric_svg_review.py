"""Fail-closed numeric receipts for editable Matplotlib SVGs.

``svg_coordinates`` accepts a polyline/polygon, an M/L/z path, or a stable
artist gid containing exactly one such path. ``svg_bars`` additionally accepts
source-linked rectangles:

    {"element_id": "bar-legacy-scored", "value": {"source": "metrics",
      "pointer": "/metrics/0/mean"}, "orientation": "horizontal",
     "baseline": 0, "category_interval": [2.69, 3.31],
     "x_scale": 2.1, "x_offset": 300, "y_scale": -12, "y_offset": 180,
     "axis_id": "matplotlib.axis_9", "category_label": "Legacy scored"}

Affine coefficients describe SVG coordinates, not data. For bars, the numeric
axis must be covered by an ``axis_layout`` decoration with matching dimension,
scale and offset. Editable numeric tick labels and their actual tick-marker
positions independently validate that affine mapping. At least two distinct
ticks are required. Category labels are checked on the orthogonal axis.
Export with ``mpl.rc_context({"svg.fonttype": "none"})`` *during savefig*.

Every rendered primitive/numeric text must be covered, not just supplied gids.
Non-data subtrees use ``decorations`` entries with element_id, sha256 (from
subtree_sha256), kind, reviewer, rationale. Kinds: background (an axis-aligned
rectangle only), axis_frame (straight Matplotlib spine), axis_layout (Matplotlib axis group only), legend (Matplotlib
legend group only), annotation (editable, nonnumeric text only), and
reviewed_illustration (also limitations and quantitative_scope=illustrative_only).
A reviewed illustration must be a separate, non-overlapping region. It never
discharges an unbound bar inside a quantitatively checked axes group.

``labels`` maps stable text/group gids to source expressions. Numeric title,
sample-size and bar annotations must be bound there. Their sources must occur
in the receipt's hash-pinned inputs. Quantitative parts of a mixed figure are
verified separately; illustration regions remain explicitly human-reviewed.
No companion-vector receipt verifies unrelated PNG/PDF bytes.

Use inventory(path) to obtain candidate gids/subtree hashes (NOT approvals),
and affine_from_bounds([left, top, right, bottom], xlim, ylim) from an exported
axes clip rectangle. All coefficients are rechecked against numeric SVG ticks.
"""
import hashlib
import math
import re
import xml.etree.ElementTree as ET


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PATH_TOKEN = re.compile(r"[MmLlZz]|" + NUMBER)
GEOMETRY = {"path", "polyline", "polygon", "rect", "line", "circle", "ellipse", "use", "image"}


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def subtree_sha256(element):
    """Hash canonical XML, independent of namespace-prefix spelling."""
    canonical = ET.canonicalize(ET.tostring(element, encoding="unicode"), rewrite_prefixes=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def path_vertices(data):
    tokens, end = [], 0
    for match in PATH_TOKEN.finditer(data):
        if data[end:match.start()].strip(" \t\r\n,"):
            raise ValueError("SVG path supports only M/L/z straight segments")
        tokens.append(match[0])
        end = match.end()
    if data[end:].strip(" \t\r\n,"):
        raise ValueError("SVG path contains unsupported coordinates/commands")
    if not tokens or tokens[0] not in ("M", "m"):
        raise ValueError("SVG path must start with a move command")
    points, command, position, closed = [], None, (0., 0.), False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("M", "m", "L", "l", "Z", "z"):
            command = token
            i += 1
            if command in ("Z", "z"):
                if not points or i != len(tokens):
                    raise ValueError("SVG path must contain one closed subpath")
                closed = True
                break
            if command in ("M", "m") and points:
                raise ValueError("multiple SVG subpaths are not one numeric series")
        if command not in ("M", "m", "L", "l") or i + 1 >= len(tokens):
            raise ValueError("incomplete SVG path")
        try:
            x, y = float(tokens[i]), float(tokens[i + 1])
        except ValueError as exc:
            raise ValueError("incomplete SVG coordinate pair") from exc
        if command.islower():
            x, y = x + position[0], y + position[1]
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("nonfinite SVG coordinate")
        points.append((x, y))
        position = (x, y)
        i += 2
        if command in ("M", "m"):
            command = "L" if command == "M" else "l"
    if not points:
        raise ValueError("empty SVG path")
    if closed and points[-1] != points[0]:
        points.append(points[0])
    return points, closed


def affine_from_bounds(bounds, xlim, ylim):
    left, top, right, bottom = bounds
    xmin, xmax = xlim
    ymin, ymax = ylim
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in [*bounds, *xlim, *ylim]):
        raise ValueError("SVG axis bounds must be finite")
    if not left < right or not top < bottom or xmin == xmax or ymin == ymax:
        raise ValueError("SVG axis bounds must have nonzero extent")
    xs, ys = (right - left) / (xmax - xmin), -(bottom - top) / (ymax - ymin)
    return {"x_scale": xs, "x_offset": left - xmin * xs,
            "y_scale": ys, "y_offset": bottom - ymin * ys}


def inventory(path):
    root = ET.fromstring(path.read_bytes())
    clips = {}
    elements = []
    for element in root.iter():
        if tag(element) == "clipPath":
            rectangles = [e for e in element if tag(e) == "rect"]
            if len(rectangles) == 1:
                x, y, w, h = [float(rectangles[0].get(k, "0")) for k in ("x", "y", "width", "height")]
                clips[element.get("id")] = [x, y, x + w, y + h]
        if element.get("id"):
            elements.append({"element_id": element.get("id"), "tag": tag(element),
                             "sha256": subtree_sha256(element)})
    return {"status": "inventory_only_not_numeric_verification", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clip_boxes": clips, "elements": elements}


def points_of(element):
    if tag(element) == "path":
        return path_vertices(element.get("d", ""))
    if tag(element) in ("polyline", "polygon"):
        data = element.get("points", "")
        return path_vertices("M " + data + (" z" if tag(element) == "polygon" else ""))
    raise ValueError("unsupported numeric SVG primitive")


def close(actual, expected, message):
    if len(actual) != len(expected) or any(
            len(a) != len(b) or any(not math.isclose(x, y, rel_tol=0, abs_tol=1e-6) for x, y in zip(a, b))
            for a, b in zip(actual, expected)):
        raise ValueError(message + "; observed=%s; expected=%s" % (actual, expected))


def reference_names(expression):
    if isinstance(expression, dict):
        result = {expression["source"]} if "source" in expression else set()
        for value in expression.values():
            result |= reference_names(value)
        return result
    if isinstance(expression, list):
        return set().union(*(reference_names(value) for value in expression)) if expression else set()
    return set()


def verify(path, validation, evidence, inputs):
    for item in inputs:
        if not {"source", "pointer", "sha256"} <= set(item):
            raise ValueError("SVG inputs require exact source pointers and hashes")
        evidence.evaluate({"source": item["source"], "pointer": item["pointer"]})
        if item["sha256"] != evidence.hashes[item["source"]]:
            raise ValueError("SVG evidence source hash changed")
    from_names = {item["source"] for item in inputs}
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as exc:
        raise ValueError("invalid SVG XML") from exc
    parents = {child: parent for parent in root.iter() for child in parent}
    elements = list(root.iter())
    ids = {}
    for element in elements:
        if element.get("id"):
            if element.get("id") in ids:
                raise ValueError("ambiguous duplicate SVG gid: " + element.get("id"))
            ids[element.get("id")] = element
    invisible = set()
    for element in elements:
        if tag(element) in ("script", "foreignObject"):
            raise ValueError("SVG executable/foreign content needs dedicated validation")
        if tag(element) == "style":
            properties = re.findall(r"([a-z-]+)\s*:", "".join(element.itertext()).lower())
            if set(properties) - {"stroke-linejoin", "stroke-linecap"}:
                raise ValueError("SVG stylesheet can alter data visibility or geometry")
        if tag(element) in ("defs", "metadata", "title", "desc"):
            invisible.update(element.iter())

    def ancestors(element):
        while element in parents:
            element = parents[element]
            yield element

    def selected(gid):
        if gid not in ids or ids[gid] in invisible:
            raise ValueError("missing SVG artist gid: " + str(gid))
        return ids[gid]

    def descendants(element):
        return [e for e in element.iter() if e not in invisible]

    def transform_free(element):
        for node in [element, *ancestors(element)]:
            style = node.get("style", "").lower()
            if node.get("transform") or "transform" in style:
                raise ValueError("transformed numeric SVG artist needs a dedicated validator")
            if node.get("mask") or node.get("filter") or re.search(r"(?:^|;)\s*(?:mask|filter)\s*:", style):
                raise ValueError("masked/filtered numeric SVG artist needs a dedicated validator")
            opacity_match = re.search(r"(?:^|;)\s*opacity\s*:\s*([^;]+)", style)
            opacity = opacity_match[1].strip() if opacity_match else node.get("opacity")
            if opacity is not None:
                try:
                    opacity_value = float(opacity.rstrip("%"))
                except ValueError as exc:
                    raise ValueError("unsupported SVG opacity declaration") from exc
                if not math.isfinite(opacity_value) or opacity_value <= 0:
                    raise ValueError("hidden SVG data cannot be numerically verified")
            if (re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden|(?:^|;)\s*opacity\s*:\s*0(?:\.0*)?(?:[;\s]|$))", style)
                    or node.get("display") == "none" or node.get("visibility") == "hidden"):
                raise ValueError("hidden SVG data cannot be numerically verified")

    covered, verified_labels, quantitative_axes, audited = set(), set(), set(), []

    def mark(element, allow_labels=False):
        nodes = set(descendants(element))
        overlap = nodes & covered
        if allow_labels:
            overlap -= verified_labels
        if overlap:
            raise ValueError("overlapping SVG receipt regions")
        covered.update(nodes)

    def expression(expr):
        names = reference_names(expr)
        if not names or not names <= from_names:
            raise ValueError("SVG expression references a source absent from hash-pinned inputs")
        return evidence.evaluate(expr)

    decorations = validation.get("decorations", [])
    decoration_ids = {}
    for item in decorations:
        if item["element_id"] in decoration_ids:
            raise ValueError("duplicate SVG decoration receipt")
        decoration_ids[item["element_id"]] = item

    def plain_text(element):
        nodes = [e for e in descendants(element) if tag(e) == "text"]
        if len(nodes) != 1:
            raise ValueError("SVG labels must be editable text; export svg.fonttype=none")
        return "".join(nodes[0].itertext()).strip()

    checked_axes = {}

    def axis(gid):
        if gid in checked_axes:
            return checked_axes[gid]
        declaration = decoration_ids.get(gid, {})
        element = selected(gid)
        if declaration.get("kind") != "axis_layout" or not re.fullmatch(r"matplotlib\.axis_\d+", gid):
            raise ValueError("bar affine mapping requires a declared Matplotlib numeric axis")
        dimension = declaration.get("dimension")
        scale, offset = declaration.get("scale"), declaration.get("offset")
        if dimension not in ("x", "y") or type(scale) not in (int, float) or not math.isfinite(scale) or not scale:
            raise ValueError("invalid SVG numeric axis affine scale")
        if type(offset) not in (int, float) or not math.isfinite(offset):
            raise ValueError("invalid SVG numeric axis affine offset")
        ticks = [e for e in descendants(element) if re.fullmatch(dimension + r"tick_\d+", e.get("id", ""))]
        values = []
        for tick in ticks:
            shown = plain_text(tick).replace("\u2212", "-").replace(",", "")
            if not re.fullmatch(NUMBER, shown):
                raise ValueError("numeric SVG axis tick is not a plain finite number")
            value = float(shown)
            markers = [e for e in descendants(tick) if tag(e) == "use" and e.get(dimension) is not None]
            if not markers:
                raise ValueError("numeric SVG tick marker position missing")
            for marker in markers:
                transform_free(marker)
                if not math.isclose(float(marker.get(dimension)), value * scale + offset, rel_tol=0, abs_tol=1e-6):
                    raise ValueError("SVG tick labels contradict declared affine mapping")
            values.append(value)
        if len(set(values)) < 2:
            raise ValueError("SVG affine mapping needs at least two distinct numeric ticks")
        checked_axes[gid] = (dimension, scale, offset)
        return checked_axes[gid]

    for series in validation.get("series", []):
        element = selected(series["element_id"])
        candidates = [e for e in descendants(element) if tag(e) in GEOMETRY]
        if len(candidates) != 1 or tag(candidates[0]) not in ("path", "polyline", "polygon"):
            raise ValueError("numeric SVG artist must contain exactly one straight path")
        primitive = candidates[0]
        transform_free(primitive)
        actual, closed = points_of(primitive)
        for key in ("x_scale", "x_offset", "y_scale", "y_offset"):
            if type(series.get(key)) not in (int, float) or not math.isfinite(series[key]):
                raise ValueError("SVG affine parameters must be finite")
        if not series["x_scale"] or not series["y_scale"]:
            raise ValueError("SVG scales must be nonzero")
        if validation["method"] == "svg_bars":
            value = expression(series["value"])
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or series.get("baseline", 0) != 0:
                raise ValueError("this SVG bar validator requires finite nonnegative values and a zero baseline")
            orientation = series.get("orientation")
            dimension = "x" if orientation == "horizontal" else "y"
            if orientation not in ("horizontal", "vertical"):
                raise ValueError("SVG bar orientation must be horizontal or vertical")
            checked = axis(series["axis_id"])
            if checked != (dimension, series[dimension + "_scale"], series[dimension + "_offset"]):
                raise ValueError("bar affine parameters differ from verified axis ticks")
            low, high = series["category_interval"]
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("invalid bar category interval")
            if orientation == "horizontal":
                data_points = [(0, low), (value, low), (value, high), (0, high)]
            else:
                data_points = [(low, 0), (high, 0), (high, value), (low, value)]
            expected = [(x * series["x_scale"] + series["x_offset"], y * series["y_scale"] + series["y_offset"])
                        for x, y in data_points]
            if not closed:
                raise ValueError("SVG bar must be a closed rectangle")
            # Start corner and winding are exporter choices; the four edges
            # must still be the exact source-derived rectangle.
            vertices = actual[:-1] if actual[-1] == actual[0] else actual
            if len(vertices) != 4:
                raise ValueError("SVG bar must have exactly four rectangle corners")
            if any(vertices[i][0] != vertices[(i + 1) % 4][0] and vertices[i][1] != vertices[(i + 1) % 4][1] for i in range(4)):
                raise ValueError("SVG bar is not axis-aligned")
            close(sorted(vertices), sorted(expected), "SVG bar coordinates differ from source value")
            clip_match = re.fullmatch(r"url\(#([^)]*)\)", primitive.get("clip-path", ""))
            clip = ids.get(clip_match[1]) if clip_match else None
            rectangles = [e for e in clip.iter() if tag(e) == "rect"] if clip is not None else []
            if len(rectangles) != 1:
                raise ValueError("SVG bar requires an unambiguous rectangular axes clip")
            rectangle = rectangles[0]
            transform_free(rectangle)
            if clip.get("clipPathUnits", "userSpaceOnUse") != "userSpaceOnUse":
                raise ValueError("SVG bar clip must use exported user-space coordinates")
            left, top, width, height = [float(rectangle.get(k, "0")) for k in ("x", "y", "width", "height")]
            if any(not left - 1e-6 <= x <= left + width + 1e-6 or not top - 1e-6 <= y <= top + height + 1e-6 for x, y in actual):
                raise ValueError("SVG bar is clipped before its source-derived endpoint")
            if not series.get("category_label"):
                raise ValueError("SVG bar requires its rendered category label")
            axes_parent = next((e for e in ancestors(element) if re.fullmatch(r"axes_\d+", e.get("id", ""))), None)
            if axes_parent is None:
                raise ValueError("SVG bar must be inside a Matplotlib axes group")
            quantitative_axes.add(axes_parent)
            other = "y" if dimension == "x" else "x"
            position = ((low + high) / 2) * series[other + "_scale"] + series[other + "_offset"]
            matching = []
            for tick in descendants(axes_parent):
                if re.fullmatch(other + r"tick_\d+", tick.get("id", "")):
                    markers = [e for e in descendants(tick) if tag(e) == "use" and e.get(other) is not None]
                    if any(math.isclose(float(e.get(other)), position, rel_tol=0, abs_tol=1e-6) for e in markers):
                        matching.append(re.sub(r"\s+", " ", " ".join("".join(e.itertext()) for e in descendants(tick) if tag(e) == "text")).strip())
            # Multiline tick labels are separate <text> elements.
            wanted = re.sub(r"\s+", " ", series["category_label"]).strip()
            if len(matching) != 1 or matching[0] != wanted:
                raise ValueError("SVG bar category label/position differs from receipt")
            audited.append({"element_id": series["element_id"], "value": value,
                            "value_expression": series["value"], "observed_vertices": actual})
        else:
            x, y = expression(series["x"]), expression(series["y"])
            if len(x) != len(y):
                raise ValueError("SVG source coordinate array lengths differ")
            expected = [(a * series["x_scale"] + series["x_offset"], b * series["y_scale"] + series["y_offset"])
                        for a, b in zip(x, y)]
            if closed and expected and expected[-1] != expected[0]:
                expected.append(expected[0])
            close(actual, expected, "SVG plotted coordinates differ from evidence")
            audited.append({"element_id": series["element_id"], "x_expression": series["x"],
                            "y_expression": series["y"], "observed_vertices": actual})
        mark(element)
    if not audited:
        raise ValueError("SVG numeric receipt cannot have empty series coverage")

    for gid, expr in validation.get("labels", {}).items():
        element = selected(gid)
        shown = plain_text(element)
        expected = str(expression(expr))
        if re.sub(r"\s+", "", shown).replace("\u2212", "-") != re.sub(r"\s+", "", expected):
            raise ValueError("SVG numeric label differs from source")
        mark(element)
        verified_labels.update(descendants(element))

    illustrations = []
    for declaration in decorations:
        element = selected(declaration["element_id"])
        nodes = descendants(element)
        if declaration.get("sha256") != subtree_sha256(element):
            raise ValueError("SVG decoration subtree changed since review")
        if not declaration.get("reviewer") or not declaration.get("rationale"):
            raise ValueError("SVG decoration requires reviewer and rationale")
        kind = declaration.get("kind")
        if kind == "background":
            geometry = [e for e in nodes if tag(e) in GEOMETRY]
            if len(geometry) != 1 or tag(geometry[0]) not in ("path", "polygon", "rect"):
                raise ValueError("SVG background must be a single rectangle")
            if any(e.get("clip-path") for e in geometry):
                raise ValueError("clipped data artist cannot be classified as background")
            if tag(geometry[0]) != "rect":
                pts, closed = points_of(geometry[0])
                if not closed or len(set(pts)) != 4 or len({p[0] for p in pts}) != 2 or len({p[1] for p in pts}) != 2:
                    raise ValueError("SVG background is not a rectangle")
                box = (min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts))
            else:
                x, y, w, h = [float(geometry[0].get(k, "0")) for k in ("x", "y", "width", "height")]
                box = x, y, x + w, y + h
            boxes = []
            if root.get("viewBox"):
                x, y, w, h = [float(v) for v in root.get("viewBox").split()]
                boxes.append((x, y, x + w, y + h))
            for node in elements:
                if tag(node) == "clipPath":
                    for rectangle in node:
                        if tag(rectangle) == "rect":
                            x, y, w, h = [float(rectangle.get(k, "0")) for k in ("x", "y", "width", "height")]
                            boxes.append((x, y, x + w, y + h))
            if not any(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(box, candidate)) for candidate in boxes):
                raise ValueError("SVG background does not cover the canvas or a complete axes rectangle")
        elif kind == "axis_frame":
            geometry = [e for e in nodes if tag(e) in GEOMETRY]
            if not re.fullmatch(r"patch_\d+", element.get("id", "")) or len(geometry) != 1:
                raise ValueError("axis frame must address one Matplotlib spine")
            pts, closed = points_of(geometry[0])
            if closed or len(pts) != 2 or (pts[0][0] != pts[1][0] and pts[0][1] != pts[1][1]):
                raise ValueError("axis frame is not a straight horizontal/vertical spine")
        elif kind == "axis_layout":
            if not re.fullmatch(r"matplotlib\.axis_\d+", element.get("id", "")):
                raise ValueError("axis-layout review must address a Matplotlib axis group")
            if declaration.get("dimension"):
                axis(declaration["element_id"])
            if any(tag(e) in ("image", "polygon", "polyline", "rect") for e in nodes):
                raise ValueError("uncovered data in SVG axis-layout region")
            for e in nodes:
                if tag(e) == "path" and points_of(e)[1]:
                    raise ValueError("closed data path cannot be an axis tick")
            tick_nodes = {e for tick in nodes if re.fullmatch(r"[xy]tick_\d+", tick.get("id", "")) for e in tick.iter()}
            if any(e not in tick_nodes and e not in verified_labels and tag(e) == "text" and re.search(NUMBER, "".join(e.itertext())) for e in nodes):
                raise ValueError("numeric axis title requires a source binding")
        elif kind == "legend":
            if not re.fullmatch(r"legend_\d+", element.get("id", "")):
                raise ValueError("legend review must address a Matplotlib legend group")
            if any(e.get("clip-path") for e in nodes if tag(e) in GEOMETRY):
                raise ValueError("clipped data artists cannot be classified as legend")
            if any(re.search(NUMBER, "".join(e.itertext())) for e in nodes if tag(e) == "text" and e not in verified_labels):
                raise ValueError("numeric legend labels require source bindings")
        elif kind == "annotation":
            if any(tag(e) in GEOMETRY for e in nodes) or any(re.search(NUMBER, "".join(e.itertext())) for e in nodes if tag(e) == "text" and e not in verified_labels):
                raise ValueError("annotation review cannot discharge data geometry/numeric text")
        elif kind == "reviewed_illustration":
            if any(element in list(e.iter()) or e in list(element.iter()) for e in quantitative_axes):
                raise ValueError("illustration review cannot discharge unbound bars in quantitative axes")
            if not declaration.get("limitations") or declaration.get("quantitative_scope") != "illustrative_only":
                raise ValueError("illustrative region needs explicit limitations and scope")
            illustrations.append({"element_id": declaration["element_id"], "status": "reviewed_illustration",
                                  "mathematically_verified": False, "limitations": declaration["limitations"]})
        else:
            raise ValueError("unsupported SVG decoration classification")
        mark(element, allow_labels=True)

    uncovered = []
    for element in elements:
        if element in invisible or element in covered:
            continue
        if tag(element) in GEOMETRY or (tag(element) == "text" and re.search(NUMBER, "".join(element.itertext()))):
            uncovered.append(element.get("id", tag(element)))
        if tag(element) in ("script", "foreignObject", "style"):
            raise ValueError("SVG executable/foreign content or global CSS needs dedicated validation")
    if uncovered:
        raise ValueError("uncovered SVG plotted series or numeric labels: " + ", ".join(uncovered[:12]))
    for item in inputs:
        current = hashlib.sha256(evidence.paths[item["source"]].read_bytes()).hexdigest()
        if current != item["sha256"]:
            raise ValueError("SVG evidence source changed during verification")
    for series in audited:
        series.update(status="programmatically_verified_plotted_values", mathematically_verified=True)
    return {"status": ("programmatically_verified_plotted_values_with_reviewed_illustration" if illustrations
                       else "programmatically_verified_plotted_values"),
            "mathematically_verified": not illustrations, "verified_series": audited,
            "illustration_regions": illustrations,
            "scope": "exact SVG series and bound numeric labels; reviewed regions are not mathematical verification"}
