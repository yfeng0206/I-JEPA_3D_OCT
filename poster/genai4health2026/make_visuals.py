"""Separate poster schematics on the attributed external OCTDL image.

The rectangular panels show full proposals before token equalization, reconstructed
from the verified replay's sizes and row-major prefixes. They are not delivered
training masks. Paper figures, replay evidence and numeric outputs are read only.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIS = ROOT / "autopilot/investigations/camera_ready_20261008/figures_v2"
sys.path.insert(0, str(VIS))
import figstyle_v2 as st
import generate_strategies_v2 as original


def main():
    staged = json.loads(original.STAGED_EVIDENCE.read_text(encoding="utf-8"))
    assert st.sha(original.STAGED_EVIDENCE) == original.STAGED_EVIDENCE_SHA256
    native, image = st.load_crop()
    _, grids, _, grid_hashes = original.guide_grids(image, staged)
    soft = (grids["soft"][0, 2] + grids["soft"][0, 3]).numpy()
    guides = {a: np.asarray(staged["guide_overlays"][a], bool) for a, _, _ in original.ARMS}
    proposals, rectangles = {}, {}
    for arm, _, _ in original.ARMS:
        groups = staged["masks"][arm]["targets"]
        if arm == "anatomy":
            proposals[arm] = [sorted(set(g)) for g in groups]
            continue
        proposals[arm], rectangles[arm] = [], []
        for group, (h, w) in zip(groups, staged["block_sizes"]["pred"]):
            row, col = divmod(min(group), 16)
            assert row + h <= 16 and col + w <= 16
            full = [(row + dy) * 16 + col + dx for dy in range(h) for dx in range(w)]
            assert group == full[:len(group)], (arm, "not a rectangle prefix")
            proposals[arm].append(full)
            rectangles[arm].append({"row": row, "col": col, "height": h, "width": w})

    out = VIS / "poster"
    layouts = {}
    outputs = {}

    def save(fig, stem):
        fig.canvas.draw()
        guard = st.layout_guard(fig, pad_pt=.5, pad_y_pt=.15, margin_pt=.2)
        assert not guard["overlaps"] and not guard["outside_canvas"], guard
        layouts[stem] = guard
        for extension in ("pdf", "png"):
            path = out / (stem + "." + extension)
            fig.savefig(path, dpi=300, facecolor="white",
                        metadata={"Creator": "OCT I-JEPA poster schematic", "CreationDate": None}
                        if extension == "pdf" else {"Description": "Illustrative schematic. OCTDL Fig. 1, Kulyabin et al.; CC BY 4.0; cropped, overlays added."})
            outputs[str(path.relative_to(ROOT)).replace("\\", "/")] = st.sha(path)
        plt.close(fig)

    with plt.rc_context(st.rc(scale=2.0, salt="poster_visual_refresh_20261025")):
        width, height, size, left, gap = 12., 6.75, 2.12, .60, .13
        fig = plt.figure(figsize=(width, height), facecolor="white")
        for j, (arm, title, subtitle) in enumerate(original.ARMS):
            x = left + j * (size + gap)
            fig.text((x + size / 2) / width, .967, title, ha="center", va="center", fontsize=16, fontweight="bold")
            fig.text((x + size / 2) / width, .924, subtitle, ha="center", va="center", fontsize=13)
            top, bottom = 4.00, 1.65
            axes = []
            ax = fig.add_axes([x / width, top / height, size / width, size / height])
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(0, 16, 16, 0))
            if arm in ("centroid", "envelope"):
                for r, c in zip(*np.nonzero(guides[arm])):
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor=st.rgba(st.GUIDE_COLOR, .42), edgecolor="none"))
                st.draw_outline(ax, guides[arm], st.GUIDE_EDGE, 1.1, halo="black", halo_lw=2.6)
            elif arm in ("anatomy", "cover"):
                for r, c in zip(*np.nonzero(soft > 0)):
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor=st.rgba(st.GUIDE_COLOR, .8 * min(float(soft[r, c]), 1)), edgecolor="none"))
            axes.append(ax)
            ax = fig.add_axes([x / width, bottom / height, size / width, size / height])
            union = set().union(*map(set, proposals[arm]))
            rgb, ctx, targets = st.ijepa_view(image, proposals[arm], sorted(set(range(256)) - union))
            assert np.all(ctx | targets)
            ax.imshow(rgb, interpolation="nearest", extent=(0, 16, 16, 0))
            for k, group in enumerate(proposals[arm]):
                st.draw_outline(ax, st.cells_mask(group), st.TARGET_COLORS[k], 1.7, halo="white", halo_lw=3.2, zorder=6+k)
            anchors, leaders, _ = st.place_numerals(proposals[arm])
            st.draw_numerals(ax, anchors, leaders, 18, 2.8, 1.1)
            axes.append(ax)
            for ax in axes:
                ax.set_xlim(0, 16); ax.set_ylim(16, 0); ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#777777"); spine.set_linewidth(.6)
            rules = {"random": "Rectangles\nuniform placement", "centroid": "Rectangles\nband weighting",
                     "envelope": "Rectangles\nretinal-overlap rule", "anatomy": "Tissue-shaped\ncomponents",
                     "cover": "Rectangles\ngreedy coverage"}
            fig.text((x+size/2)/width, 1.48/height, rules[arm], ha="center", va="top", fontsize=14, linespacing=1.2)
        for y, label in ((5.06, "Placement guide"), (2.71, "Target proposals")):
            fig.text(.021, y/height, label, rotation=90, ha="center", va="center", fontsize=14, fontweight="bold")
        for k, color in enumerate(st.TARGET_COLORS):
            x = .105 + k*.038
            fig.add_artist(Rectangle((x,.078),.027,.037,transform=fig.transFigure,facecolor=color,edgecolor="#333333",linewidth=.5))
            fig.text(x+.0135,.097,str(k+1),ha="center",va="center",fontsize=14,fontweight="bold")
        fig.text(.27,.097,"target groups; overlap allowed",va="center",fontsize=14)
        fig.text(.66,.097,"Yellow: placement guide",va="center",fontsize=14)
        fig.text(.5,.049,"Illustrative proposals BEFORE batching and token equalization",ha="center",va="center",fontsize=14,fontweight="bold",color="#103453")
        fig.text(.5,.015,"OCT: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0), cropped; not FairVision.",ha="center",va="center",fontsize=12)
        save(fig, "fig_oct_proposals_poster")

        fig = plt.figure(figsize=(8.6,3.45),facecolor="white")
        ax = fig.add_axes([.015,.035,.49,.93])
        ax.imshow(native,cmap="gray",vmin=0,vmax=255,interpolation="nearest",extent=(0,16,16,0))
        ax.set_xlim(0,16);ax.set_ylim(16,0);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        # Anatomical region labels, not clinical layer annotations or a diagnosis.
        for text, xy, position, color in [
            ("Low-intensity\nbackground",(11,2.4),(1.23,.77),"#0072B2"),
            ("Retinal tissue\nband",(11,8.3),(1.23,.43),"#8C7F00")]:
            ax.annotate(text,xy=xy,xycoords="data",xytext=position,textcoords="axes fraction",
                        ha="left",va="center",fontsize=20,fontweight="bold",color="#103453",
                        arrowprops=dict(arrowstyle="->",color=color,lw=2,connectionstyle="arc3,rad=0"),
                        annotation_clip=False)
        fig.text(.583,.085,"One B-scan; structure\nvaries across the volume.",fontsize=16,ha="left",va="bottom",color="#444444",linespacing=1.3)
        save(fig, "fig_oct_structure_poster")

    evidence = {"kind":"poster_visual_schematics_v1", "generator":"poster/genai4health2026/make_visuals.py",
                "generator_sha256":st.sha(Path(__file__)), "source_image_sha256":st.SOURCE_SHA256,
                "source_crop":list(st.CROP), "attribution":st.ATTRIBUTION,
                "license":"CC BY 4.0", "license_url":"https://creativecommons.org/licenses/by/4.0/",
                "source_url":"https://arxiv.org/abs/2312.08255v3",
                "stage":"Illustrative target proposals before token equalization; not delivered training masks",
                "replay_source_sha256":st.sha(original.STAGED_EVIDENCE), "guide_grids_sha256":grid_hashes,
                "rectangles":rectangles,"anatomy":"Unique component cells from the staged replay; repetitions omitted",
                "structure_image":"Same external OCTDL crop at native crop resolution, no contrast enhancement; region labels are schematic",
                "clinical_images":"External OCTDL example only; no FairVision pixels or patient identifiers",
                "layouts":layouts,"outputs":outputs}
    path = out / "poster_visual_schematics.evidence.json"
    path.write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    registry_path = ROOT / "autopilot/release_assets.json"
    registry_text=registry_path.read_text(encoding="utf-8")
    registry=json.loads(registry_text)
    registry["poster_visual_schematics"]={"script":evidence["generator"],"script_sha256":evidence["generator_sha256"],
        "evidence_manifest":str(path.relative_to(ROOT)).replace("\\","/"),"evidence_manifest_sha256":st.sha(path),
        "scope":evidence["stage"],"license":"CC BY 4.0","poster_variant":outputs}
    # The poster-only entry is last. Preserve all existing paper registry text.
    if '\n  "poster_visual_schematics": ' in registry_text:
        prefix=registry_text.split('\n  "poster_visual_schematics": ',1)[0].rstrip()
        assert prefix.endswith(',')
        prefix=prefix[:-1]
    else:
        prefix=registry_text.rstrip()[:-1].rstrip()
    assert json.loads(prefix+'\n}\n') == {k:v for k,v in registry.items() if k!='poster_visual_schematics'}
    entry='  "poster_visual_schematics": '+json.dumps(registry['poster_visual_schematics'],indent=2).replace('\n','\n  ')
    registry_path.write_text(prefix+',\n'+entry+'\n}\n',encoding="utf-8")
    print("Generated and registered clean proposals + annotated OCT structure; original figures unchanged.")


if __name__ == "__main__":
    main()
