import random
from dataclasses import asdict, dataclass, field


@dataclass
class CircuitSample:
    family: str
    prompt: str
    target: str
    expected_supernodes: list[tuple[str, list[str]]] = field(default_factory=list)
    expected_edges: list[tuple[str, str]] = field(default_factory=list)

    def to_row(self) -> dict:
        return asdict(self)


TWO_HOP = [
    # (city, region, capital, region_kind)
    ("Dallas", "Texas", "Austin", "state"),
    ("Oakland", "California", "Sacramento", "state"),
    ("Savannah", "Georgia", "Atlanta", "state"),
    ("Miami", "Florida", "Tallahassee", "state"),
    ("Cleveland", "Ohio", "Columbus", "state"),
    ("Vancouver", "British Columbia", "Victoria", "province"),
    ("Shanghai", "China", "Beijing", "country"),
    ("Munich", "Germany", "Berlin", "country"),
    ("Osaka", "Japan", "Tokyo", "country"),
    ("Marseille", "France", "Paris", "country"),
]

ANTONYMS = {
    "en": [
        ("hot", "cold"),
        ("small", "big"),
        ("fast", "slow"),
        ("dark", "light"),
        ("happy", "sad"),
        ("tall", "short"),
        ("early", "late"),
        ("wet", "dry"),
    ],
    "fr": [
        ("petit", "grand"),
        ("chaud", "froid"),
        ("rapide", "lent"),
        ("jeune", "vieux"),
    ],
    "es": [("grande", "pequeño"), ("caliente", "frío"), ("alto", "bajo")],
}
ANTONYM_TEMPLATES = {
    "en": 'The opposite of "{w}" is "',
    "fr": 'Le contraire de "{w}" est "',
    "es": 'Lo contrario de "{w}" es "',
}

ANALOGIES = [
    # (a, b, c, d, relation)
    ("Mexico", "peso", "Europe", "euro", "currency-of"),
    ("Japan", "yen", "India", "rupee", "currency-of"),
    ("Zagreb", "Croatia", "Copenhagen", "Denmark", "capital-to-country"),
    ("Lisbon", "Portugal", "Helsinki", "Finland", "capital-to-country"),
    ("puppy", "dog", "kitten", "cat", "young-of"),
    ("France", "French", "Germany", "German", "language-of"),
]

ACRONYM_WORDS = {
    "adj1": ["National", "International", "Global", "United", "Advanced", "Central"],
    "adj2": ["Digital", "Security", "Research", "Analytics", "Development", "Trade"],
    "noun": ["Analytics", "Systems", "Standards", "Operations", "Resources", "Policy"],
    "suffix": ["Group", "Agency", "Bureau", "Council", "Institute", "Office"],
}

FACTUAL_SPORTS = [
    ("Michael Jordan", "basketball"),
    ("Lionel Messi", "soccer"),
    ("Serena Williams", "tennis"),
    ("Tiger Woods", "golf"),
    ("Babe Ruth", "baseball"),
    ("Wayne Gretzky", "hockey"),
    ("Usain Bolt", "sprinting"),
    ("Tom Brady", "football"),
]

INCONTEXT_ATTRS = [
    ("grass", "green"),
    ("sky", "blue"),
    ("corn", "yellow"),
    ("carrot", "orange"),
    ("strawberry", "red"),
    ("snow", "white"),
    ("coal", "black"),
    ("grape", "purple"),
]


def _two_hop(rng: random.Random) -> CircuitSample:
    city, region, capital, kind = rng.choice(TWO_HOP)
    # one-shot exemplar: Qwen3-4B answers " which" to the bare zero-shot prompt
    ec, _, ecap, ekind = rng.choice([t for t in TWO_HOP if t[0] != city])
    return CircuitSample(
        family="two_hop",
        prompt=(
            f"Fact: the capital of the {ekind} containing {ec} is {ecap}.\n"
            f"Fact: the capital of the {kind} containing {city} is"
        ),
        target=f" {capital}",
        expected_supernodes=[
            ("city", [city]),
            ("capital-relation", ["capital"]),
            ("region", [region]),
            (f"say-{capital}", [capital, "say", "capital"]),
        ],
        expected_edges=[
            ("city", "region"),
            ("region", f"say-{capital}"),
            ("capital-relation", f"say-{capital}"),
        ],
    )


def _addition(rng: random.Random) -> CircuitSample:
    a, b = rng.randint(12, 89), rng.randint(12, 89)
    s = a + b
    ones = s % 10
    # one-shot exemplar: Qwen3-4B answers "?" to a bare "calc:" prompt
    ea, eb = rng.randint(12, 89), rng.randint(12, 89)
    while (ea, eb) == (a, b):
        ea, eb = rng.randint(12, 89), rng.randint(12, 89)
    return CircuitSample(
        family="addition",
        prompt=f"calc: {ea}+{eb}={ea + eb}\ncalc: {a}+{b}=",
        target=str(s),
        expected_supernodes=[
            ("operand-a", [str(a)]),
            ("operand-b", [str(b)]),
            ("ones-digit", [f"_{ones}", f"ends in {ones}", str(ones)]),
            ("magnitude", ["sum", "~" + str(round(s, -1))]),
            (f"say-{s}", [str(s), "say"]),
        ],
        expected_edges=[
            ("operand-a", "ones-digit"),
            ("operand-b", "ones-digit"),
            ("operand-a", "magnitude"),
            ("operand-b", "magnitude"),
            ("ones-digit", f"say-{s}"),
            ("magnitude", f"say-{s}"),
        ],
    )


def _antonym(rng: random.Random) -> CircuitSample:
    lang = rng.choice(list(ANTONYMS))
    w, ant = rng.choice(ANTONYMS[lang])
    ew, ea = rng.choice([p for p in ANTONYMS[lang] if p[0] != w])
    exemplar = ANTONYM_TEMPLATES[lang].format(w=ew) + ea + '".\n'
    return CircuitSample(
        family="antonym",
        prompt=exemplar + ANTONYM_TEMPLATES[lang].format(w=w),
        target=ant,
        expected_supernodes=[
            ("operand-word", [w]),
            ("antonym-relation", ["opposite", "antonym", "contraire", "contrario"]),
            (f"say-{ant}", [ant, "say"]),
        ],
        expected_edges=[
            ("operand-word", f"say-{ant}"),
            ("antonym-relation", f"say-{ant}"),
        ],
    )


def _analogy(rng: random.Random) -> CircuitSample:
    a, b, c, d, rel = rng.choice(ANALOGIES)
    ex = rng.choice(
        [x for x in ANALOGIES if x[4] == rel and x[0] != a]
        or [x for x in ANALOGIES if x[0] != a]
    )
    return CircuitSample(
        family="analogy",
        prompt=f"{ex[0]}:{ex[1]} :: {ex[2]}:{ex[3]}\n{a}:{b} :: {c}:",
        target=d,
        expected_supernodes=[
            ("exemplar-pair", [a, b]),
            ("relation", [rel]),
            ("query", [c]),
            (f"say-{d}", [d, "say"]),
        ],
        expected_edges=[
            ("exemplar-pair", "relation"),
            ("relation", f"say-{d}"),
            ("query", f"say-{d}"),
        ],
    )


def _acronym(rng: random.Random) -> CircuitSample:
    w1 = rng.choice(ACRONYM_WORDS["adj1"])
    w2 = rng.choice(ACRONYM_WORDS["adj2"])
    w3 = rng.choice([w for w in ACRONYM_WORDS["noun"] if w != w2])
    suf = rng.choice(ACRONYM_WORDS["suffix"])
    acronym = w1[0] + w2[0] + w3[0] + suf[0]
    return CircuitSample(
        family="acronym",
        prompt=f"The {w1} {w2} {w3} {suf} ({acronym[0]}",
        target=acronym[1:],
        expected_supernodes=[
            ("word-initials", [w1, w2, w3, suf, "first letter", "initial"]),
            ("acronym-mode", ["acronym", "abbreviation", "caps"]),
            (f"say-{acronym[1:]}", [acronym[1:], "say"]),
        ],
        expected_edges=[
            ("word-initials", f"say-{acronym[1:]}"),
            ("acronym-mode", f"say-{acronym[1:]}"),
        ],
    )


def _sequence(rng: random.Random) -> CircuitSample:
    step = rng.choice([3, 4, 6, 7, 9, 11])
    start = step * rng.randint(1, 4)
    terms = [start + i * step for i in range(5)]
    nxt = terms[-1] + step
    return CircuitSample(
        family="sequence",
        # trailing space: Qwen tokenizes " 42" as [" ", "4", "2"], so without it
        # the top predicted token is always the bare space and faithfulness fails
        prompt=" ".join(str(t) for t in terms) + " ",
        target=str(nxt),
        expected_supernodes=[
            ("sequence-terms", [str(t) for t in terms]),
            ("step-pattern", [f"+{step}", "multiples", str(step)]),
            (f"say-{nxt}", [str(nxt), "say"]),
        ],
        expected_edges=[
            ("sequence-terms", "step-pattern"),
            ("step-pattern", f"say-{nxt}"),
        ],
    )


def _factual(rng: random.Random) -> CircuitSample:
    person, sport = rng.choice(FACTUAL_SPORTS)
    return CircuitSample(
        family="factual",
        prompt=f"Fact: {person} plays the sport of",
        target=f" {sport}",
        expected_supernodes=[
            ("entity", person.split()),
            ("sport-relation", ["sport", "plays"]),
            (f"say-{sport}", [sport, "say"]),
        ],
        expected_edges=[("entity", f"say-{sport}"), ("sport-relation", f"say-{sport}")],
    )


def _incontext(rng: random.Random) -> CircuitSample:
    pairs = rng.sample(INCONTEXT_ATTRS, 5)
    ctx, (qk, qv) = pairs[:4], pairs[4]
    prompt = " ".join(f"{k}: {v}" for k, v in ctx) + f" {qk}:"
    return CircuitSample(
        family="incontext",
        prompt=prompt,
        target=f" {qv}",
        expected_supernodes=[
            ("pattern", ["colon", "pair", "pattern", "color"]),
            ("query-item", [qk]),
            (f"say-{qv}", [qv, "say"]),
        ],
        expected_edges=[("pattern", f"say-{qv}"), ("query-item", f"say-{qv}")],
    )


FAMILIES = {
    "two_hop": _two_hop,
    "addition": _addition,
    "antonym": _antonym,
    "analogy": _analogy,
    "acronym": _acronym,
    "sequence": _sequence,
    "factual": _factual,
    "incontext": _incontext,
}


def generate(
    n: int, seed: int = 0, families: list[str] | None = None
) -> list[CircuitSample]:
    if n < 1:
        raise ValueError("n must be >= 1")
    fams = list(FAMILIES) if families is None else list(families)
    if not fams:
        raise ValueError("families must not be empty")
    if unknown := set(fams) - FAMILIES.keys():
        raise ValueError(f"unknown families: {sorted(unknown)}")
    rng = random.Random(seed)
    base, extra = divmod(n, len(fams))
    samples: list[CircuitSample] = []
    seen: set[str] = set()
    for i, fam in enumerate(fams):
        quota = base + (i < extra)
        misses = 0
        while quota:
            s = FAMILIES[fam](rng)
            if s.prompt in seen:
                misses += 1
                if misses > 50 * (quota + 1):
                    raise ValueError(f"family {fam!r} ran out of unique prompts")
                continue
            seen.add(s.prompt)
            samples.append(s)
            quota -= 1
    rng.shuffle(samples)
    return samples
