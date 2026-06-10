from __future__ import annotations

import csv
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "related_work_matrix.csv"


@dataclass(frozen=True)
class Entry:
    title: str
    authors: str
    year: str
    venue: str
    url: str
    area: str
    relevance: str
    adversarial_note: str


SEED_ENTRIES = [
    Entry(
        "Flow Matching for Generative Modeling",
        "Yaron Lipman; Ricky T. Q. Chen; Heli Ben-Hamu; Maximilian Nickel; Matt Le",
        "2023",
        "ICLR",
        "https://arxiv.org/abs/2210.02747",
        "flow matching",
        "Introduces simulation-free CNF training by vector-field regression.",
        "Does not study inference-time reranking or world-model value exploitation.",
    ),
    Entry(
        "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow",
        "Xingchao Liu; Chengyue Gong; Qiang Liu",
        "2023",
        "ICLR",
        "https://arxiv.org/abs/2209.03003",
        "rectified flow",
        "Defines straight-line transport training that motivates the generator used here.",
        "Efficiency and straightness claims leave open downstream selection failures.",
    ),
    Entry(
        "Scaling Laws for Reward Model Overoptimization",
        "Leo Gao; John Schulman; Jacob Hilton",
        "2023",
        "ICML",
        "https://arxiv.org/abs/2210.10760",
        "Best-of-N and reward hacking",
        "Directly measures Best-of-N proxy overoptimization.",
        "Language-model setting, not continuous trajectory world models.",
    ),
    Entry(
        "Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment",
        "Yuu Jinnai; et al.",
        "2024",
        "arXiv",
        "https://arxiv.org/abs/2404.01054",
        "Best-of-N repair",
        "Shows that regularizing Best-of-N can reduce proxy exploitation.",
        "Our repair must be framed as an adaptation, not a new general principle.",
    ),
    Entry(
        "World Models",
        "David Ha; Juergen Schmidhuber",
        "2018",
        "NeurIPS workshop",
        "https://arxiv.org/abs/1803.10122",
        "world models",
        "Classic latent dynamics motivation for planning inside a model.",
        "Autoregressive latent dynamics differ from flow-generated trajectory models.",
    ),
    Entry(
        "Learning Latent Dynamics for Planning from Pixels",
        "Danijar Hafner; Timothy Lillicrap; Ian Fischer; et al.",
        "2019",
        "ICML",
        "https://arxiv.org/abs/1811.04551",
        "world models",
        "PlaNet demonstrates planning in learned latent dynamics.",
        "Does not address reranking many generative rollouts by a brittle score.",
    ),
    Entry(
        "Dream to Control: Learning Behaviors by Latent Imagination",
        "Danijar Hafner; Timothy Lillicrap; Jimmy Ba; Mohammad Norouzi",
        "2020",
        "ICLR",
        "https://arxiv.org/abs/1912.01603",
        "world models",
        "Dreamer connects learned world models to value learning.",
        "Uses imagined rollouts differently from Best-of-N whole-trajectory selection.",
    ),
    Entry(
        "Mastering Diverse Domains through World Models",
        "Danijar Hafner; Jurgis Pasukonis; Jimmy Ba; Timothy Lillicrap",
        "2023",
        "arXiv",
        "https://arxiv.org/abs/2301.04104",
        "world models",
        "DreamerV3 shows broad world-model control performance.",
        "Strong baseline family; our toy setting is not competitive with it.",
    ),
    Entry(
        "Diffuser: Diffusion Models for Offline Reinforcement Learning",
        "Michael Janner; Yilun Du; Joshua Tenenbaum; Sergey Levine",
        "2022",
        "ICML",
        "https://arxiv.org/abs/2205.09991",
        "diffusion planning",
        "Generates and optimizes trajectories with a diffusion model.",
        "Closest planning analogue; any flow claim must distinguish transport sampler behavior.",
    ),
    Entry(
        "Planning with Diffusion for Flexible Behavior Synthesis",
        "Michael Janner; Yilun Du; Joshua B. Tenenbaum; Sergey Levine",
        "2022",
        "ICML",
        "https://diffusion-planning.github.io/",
        "diffusion planning",
        "Shows planning can be cast as generative trajectory sampling.",
        "Selection and guidance issues are already central in diffusion planning.",
    ),
    Entry(
        "Is Conditional Generative Modeling all you need for Decision Making?",
        "Ajay; Du; Gupta; Tenenbaum; Jaakkola; Agrawal",
        "2023",
        "ICLR",
        "https://arxiv.org/abs/2211.15657",
        "decision diffusion",
        "Decision Diffuser conditions trajectories on returns/goals.",
        "Conditioning is not identical to post-hoc Best-of-N reranking.",
    ),
    Entry(
        "TD-MPC2: Scalable, Robust World Models for Continuous Control",
        "Nicklas Hansen; Hao Su; Xiaolong Wang",
        "2024",
        "ICLR",
        "https://arxiv.org/abs/2310.16828",
        "model-based RL",
        "Modern strong latent model-based control baseline.",
        "Trajectory flow generator here is a diagnostic toy, not a TD-MPC replacement.",
    ),
    Entry(
        "EfficientZero: Mastering Atari Games with Limited Data",
        "Weirui Ye; Shaohuai Liu; Thanard Kurutach; Pieter Abbeel; Yang Gao",
        "2021",
        "ICLR",
        "https://arxiv.org/abs/2111.00210",
        "model-based RL",
        "Shows model-based planning can be sample-efficient.",
        "Tree-search dynamics are distinct from whole-trajectory generative reranking.",
    ),
    Entry(
        "Consistency Models",
        "Yang Song; Prafulla Dhariwal; Mark Chen; Ilya Sutskever",
        "2023",
        "ICML",
        "https://arxiv.org/abs/2303.01469",
        "fast generative sampling",
        "Few-step generation matters for candidate sampling budgets.",
        "Does not directly analyze value-reranked world-model samples.",
    ),
    Entry(
        "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions",
        "Michael S. Albergo; Nicholas M. Boffi; Eric Vanden-Eijnden",
        "2023",
        "arXiv",
        "https://arxiv.org/abs/2303.08797",
        "flow/diffusion theory",
        "Places flow matching and diffusion under a shared interpolation view.",
        "Broad unification reduces room for architecture-only novelty claims.",
    ),
    Entry(
        "Building Normalizing Flows with Stochastic Interpolants",
        "Michael S. Albergo; Eric Vanden-Eijnden",
        "2023",
        "ICLR",
        "https://arxiv.org/abs/2209.15571",
        "flow/diffusion theory",
        "Another simulation-free bridge between data and base distributions.",
        "The project should not present rectified flow as the only possible formalism.",
    ),
    Entry(
        "Classifier-Free Diffusion Guidance",
        "Jonathan Ho; Tim Salimans",
        "2022",
        "NeurIPS workshop",
        "https://arxiv.org/abs/2207.12598",
        "guided generation",
        "Guidance is an important alternative to post-hoc reranking.",
        "Best-of-N should be compared conceptually to guidance and conditioning.",
    ),
    Entry(
        "Denoising Diffusion Probabilistic Models",
        "Jonathan Ho; Ajay Jain; Pieter Abbeel",
        "2020",
        "NeurIPS",
        "https://arxiv.org/abs/2006.11239",
        "diffusion foundation",
        "Diffusion sampling is the major neighboring generative baseline.",
        "Any flow-specific claim needs more than generic generative sampling issues.",
    ),
    Entry(
        "Score-Based Generative Modeling through Stochastic Differential Equations",
        "Yang Song; Jascha Sohl-Dickstein; Diederik P. Kingma; et al.",
        "2021",
        "ICLR",
        "https://arxiv.org/abs/2011.13456",
        "diffusion foundation",
        "Probability-flow ODEs connect diffusion and deterministic sampling.",
        "Weakens claims that ODE sampling diagnostics are unique to rectified flow.",
    ),
    Entry(
        "Understanding World or Predicting Future? A Comprehensive Survey of World Models",
        "Tsinghua FIB Lab authors",
        "2025",
        "ACM Computing Surveys",
        "https://github.com/tsinghua-fib-lab/World-Model",
        "world-model survey",
        "Provides a broad map of world-model variants and tasks.",
        "Useful for scope control; the repo cannot cover all world-model families.",
    ),
]


ARXIV_QUERIES = [
    ("flow matching", 'all:"flow matching"', 45),
    ("rectified flow", 'all:"rectified flow"', 35),
    ("stochastic interpolant", 'all:"stochastic interpolant"', 25),
    ("diffusion reinforcement learning", 'all:"diffusion model" AND all:"reinforcement learning"', 45),
    ("world model", 'all:"world model" AND cat:cs.LG', 45),
    ("trajectory diffusion", 'all:"trajectory diffusion"', 30),
    ("reward model overoptimization", 'all:"reward model" AND all:"overoptimization"', 20),
    ("best of n", 'all:"best-of-n" OR all:"best of n"', 20),
    ("model-based reinforcement learning", 'all:"model-based reinforcement learning"', 45),
]


def relation_for(label: str) -> tuple[str, str]:
    if "flow" in label or "interpolant" in label:
        return (
            "flow/transport generative modeling",
            "Checks whether the phenomenon is generic to transport samplers.",
        )
    if "world" in label or "model-based" in label:
        return (
            "world models and model-based control",
            "Pressures the claim against mature model-based RL alternatives.",
        )
    if "reward" in label or "best" in label:
        return (
            "proxy optimization and Best-of-N",
            "Prior reward-hacking results may already explain the failure mode.",
        )
    return (
        "diffusion and trajectory generation",
        "Neighboring generative planners may share the same selection pathology.",
    )


def fetch_arxiv(label: str, query: str, max_results: int) -> list[Entry]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read()
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries: list[Entry] = []
    area, note = relation_for(label)
    for item in root.findall("atom:entry", ns):
        title = " ".join(item.findtext("atom:title", default="", namespaces=ns).split())
        if not title:
            continue
        authors = "; ".join(
            " ".join(author.findtext("atom:name", default="", namespaces=ns).split())
            for author in item.findall("atom:author", ns)
        )
        published = item.findtext("atom:published", default="", namespaces=ns)
        year = published[:4] if published else ""
        paper_url = item.findtext("atom:id", default="", namespaces=ns)
        entries.append(
            Entry(
                title=title,
                authors=authors,
                year=year,
                venue="arXiv",
                url=paper_url,
                area=area,
                relevance=f"Retrieved by arXiv query: {label}.",
                adversarial_note=note,
            )
        )
    return entries


def build_entries() -> list[Entry]:
    entries: list[Entry] = list(SEED_ENTRIES)
    seen = {e.title.lower() for e in entries}
    for label, query, limit in ARXIV_QUERIES:
        try:
            fetched = fetch_arxiv(label, query, limit)
        except Exception as exc:
            print(f"warning: arXiv query failed for {label}: {exc}")
            fetched = []
        for entry in fetched:
            key = entry.title.lower()
            if key not in seen:
                entries.append(entry)
                seen.add(key)
        time.sleep(0.4)
    return entries


def main() -> None:
    entries = build_entries()
    if len(entries) < 100:
        raise RuntimeError(f"Only collected {len(entries)} entries; expected at least 100.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "title",
                "authors",
                "year",
                "venue",
                "url",
                "area",
                "relevance_to_project",
                "adversarial_note",
            ],
        )
        writer.writeheader()
        for i, entry in enumerate(entries, start=1):
            writer.writerow(
                {
                    "id": i,
                    "title": entry.title,
                    "authors": entry.authors,
                    "year": entry.year,
                    "venue": entry.venue,
                    "url": entry.url,
                    "area": entry.area,
                    "relevance_to_project": entry.relevance,
                    "adversarial_note": entry.adversarial_note,
                }
            )
    print(f"wrote {len(entries)} entries to {OUT}")


if __name__ == "__main__":
    main()
