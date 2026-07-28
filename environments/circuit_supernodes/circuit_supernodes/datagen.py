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
    ("Rotterdam", "Netherlands", "Amsterdam", "country"),
    ("Porto", "Portugal", "Lisbon", "country"),
    ("Krakow", "Poland", "Warsaw", "country"),
    ("Guadalajara", "Mexico", "Mexico City", "country"),
    ("Busan", "South Korea", "Seoul", "country"),
    ("Salvador", "Brazil", "Brasilia", "country"),
    ("Calgary", "Alberta", "Edmonton", "province"),
    ("Buffalo", "New York", "Albany", "state"),
    ("Portland", "Oregon", "Salem", "state"),
    ("Tucson", "Arizona", "Phoenix", "state"),
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
        ("hard", "soft"),
        ("rich", "poor"),
        ("clean", "dirty"),
        ("open", "closed"),
        ("heavy", "light"),
        ("sharp", "blunt"),
    ],
    "fr": [
        ("petit", "grand"),
        ("chaud", "froid"),
        ("rapide", "lent"),
        ("jeune", "vieux"),
        ("fort", "faible"),
        ("propre", "sale"),
        ("lourd", "léger"),
    ],
    "es": [
        ("grande", "pequeño"),
        ("caliente", "frío"),
        ("alto", "bajo"),
        ("nuevo", "viejo"),
        ("fuerte", "débil"),
    ],
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
    ("Brazil", "real", "Poland", "zloty", "currency-of"),
    ("Norway", "krone", "Sweden", "krona", "currency-of"),
    ("Ottawa", "Canada", "Canberra", "Australia", "capital-to-country"),
    ("Vienna", "Austria", "Athens", "Greece", "capital-to-country"),
    ("calf", "cow", "foal", "horse", "young-of"),
    ("cub", "bear", "chick", "chicken", "young-of"),
    ("Italy", "Italian", "Portugal", "Portuguese", "language-of"),
    ("Japan", "Japanese", "Korea", "Korean", "language-of"),
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
    ("Roger Federer", "tennis"),
    ("Cristiano Ronaldo", "soccer"),
    ("LeBron James", "basketball"),
    ("Muhammad Ali", "boxing"),
    ("Michael Phelps", "swimming"),
    ("Simone Biles", "gymnastics"),
    ("Rory McIlroy", "golf"),
    ("Sidney Crosby", "hockey"),
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
    ("lime", "green"),
    ("ash", "grey"),
    ("cherry", "red"),
    ("sand", "beige"),
    ("ocean", "blue"),
    ("banana", "yellow"),
    ("chocolate", "brown"),
    ("pearl", "white"),
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
    step = rng.choice([3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17])
    start = step * rng.randint(1, 9)
    terms = [start + i * step for i in range(rng.choice([4, 5, 6]))]
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


FACTUAL_FRAMES = [
    "Fact: {person} plays the sport of",
    "{person} competes professionally in the sport of",
    "The athlete {person} is famous for playing",
    "Trivia: the sport most associated with {person} is",
    "{person} built a career in the sport of",
]


def _factual(rng: random.Random) -> CircuitSample:
    person, sport = rng.choice(FACTUAL_SPORTS)
    return CircuitSample(
        family="factual",
        prompt=rng.choice(FACTUAL_FRAMES).format(person=person),
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


# --- sealed OOD families -------------------------------------------------
#
# Held back for the final evaluation and never used for training, tuning or
# inspection. Each is built so the answer cannot be reached by a surface
# shortcut: the agreement prompts put an attractor noun of the opposite number
# between subject and verb, the comparisons are balanced across both answers and
# across whether the larger number appears first, and the weekday arithmetic
# wraps the week so the offset cannot be read off the ordering.

SVA_SUBJECTS = [
    # (singular, plural, attractor-singular, attractor-plural)
    ("key", "keys", "cabinet", "cabinets"),
    ("author", "authors", "novel", "novels"),
    ("pilot", "pilots", "aircraft", "aircraft"),
    ("senator", "senators", "committee", "committees"),
    ("painting", "paintings", "gallery", "galleries"),
    ("engineer", "engineers", "server", "servers"),
    ("farmer", "farmers", "tractor", "tractors"),
    ("surgeon", "surgeons", "clinic", "clinics"),
    ("teacher", "teachers", "classroom", "classrooms"),
    ("dancer", "dancers", "rehearsal", "rehearsals"),
    ("architect", "architects", "blueprint", "blueprints"),
    ("chemist", "chemists", "reagent", "reagents"),
    ("banker", "bankers", "ledger", "ledgers"),
    ("captain", "captains", "vessel", "vessels"),
    ("journalist", "journalists", "archive", "archives"),
    ("gardener", "gardeners", "greenhouse", "greenhouses"),
    ("mechanic", "mechanics", "workshop", "workshops"),
    ("violinist", "violinists", "recital", "recitals"),
]

# The attractor has to sit between subject and verb, so the phrase varies too —
# ten subjects alone give only twenty prompts, which is memorisation bait at RL
# visit counts.
SVA_PHRASES = [
    "near the",
    "beside the",
    "behind the",
    "across from the",
    "in front of the",
    "next to the",
    "opposite the",
    "beyond the",
]

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _agreement(rng: random.Random) -> CircuitSample:
    """Subject-verb agreement with a number-mismatched attractor in between."""
    sing, plur, attr_s, attr_p = rng.choice(SVA_SUBJECTS)
    subject_plural = rng.random() < 0.5
    subject = plur if subject_plural else sing
    attractor = attr_s if subject_plural else attr_p  # deliberately mismatched
    verb = "are" if subject_plural else "is"
    phrase = rng.choice(SVA_PHRASES)
    return CircuitSample(
        family="agreement",
        prompt=f"The {subject} {phrase} {attractor}",
        target=f" {verb}",
        expected_supernodes=[
            ("subject-number", [subject, "plural" if subject_plural else "singular"]),
            ("attractor", [attractor]),
            (f"say-{verb}", [verb, "say"]),
        ],
        expected_edges=[
            ("subject-number", f"say-{verb}"),
            ("attractor", f"say-{verb}"),
        ],
    )


def _comparison(rng: random.Random) -> CircuitSample:
    """Two-digit comparison, balanced over the answer AND over argument order.

    With a single relation the two are perfectly correlated — "is X greater than
    Y" answers Yes exactly when the larger number comes first — so a model can
    score without comparing anything. Varying the relation decorrelates them:
    "is 27 less than 63" is also Yes, with the smaller number first.
    """
    want_yes = rng.random() < 0.5
    greater = rng.random() < 0.5
    high, low = sorted(rng.sample(range(10, 100), 2), reverse=True)
    # "greater" is true iff the larger is first; "less" iff the smaller is.
    larger_first = greater == want_yes
    first, second = (high, low) if larger_first else (low, high)
    relation = "greater" if greater else "less"
    answer = "Yes" if want_yes else "No"
    return CircuitSample(
        family="comparison",
        prompt=(
            f"Q: Is 63 greater than 27? A: Yes\n"
            f"Q: Is {first} {relation} than {second}? A:"
        ),
        target=f" {answer}",
        expected_supernodes=[
            ("left-operand", [str(first)]),
            ("right-operand", [str(second)]),
            ("comparison-relation", [relation, "compare", "than"]),
            (f"say-{answer}", [answer, "say"]),
        ],
        expected_edges=[
            ("left-operand", f"say-{answer}"),
            ("right-operand", f"say-{answer}"),
            ("comparison-relation", f"say-{answer}"),
        ],
    )


def _weekday(rng: random.Random) -> CircuitSample:
    """Day-of-week arithmetic, wrapping so the offset is not readable in order."""
    start = rng.randrange(len(WEEKDAYS))
    offset = rng.randint(2, 6)
    answer = WEEKDAYS[(start + offset) % len(WEEKDAYS)]
    ex_start = rng.randrange(len(WEEKDAYS))
    ex_offset = rng.randint(2, 6)
    ex_answer = WEEKDAYS[(ex_start + ex_offset) % len(WEEKDAYS)]
    return CircuitSample(
        family="weekday",
        prompt=(
            f"{ex_offset} days after {WEEKDAYS[ex_start]} is {ex_answer}.\n"
            f"{offset} days after {WEEKDAYS[start]} is"
        ),
        target=f" {answer}",
        expected_supernodes=[
            ("start-day", [WEEKDAYS[start]]),
            ("offset", [str(offset), "days after"]),
            (f"say-{answer}", [answer, "say"]),
        ],
        expected_edges=[("start-day", f"say-{answer}"), ("offset", f"say-{answer}")],
    )


SEALED_FAMILIES = ("agreement", "comparison", "weekday")

FAMILIES = {
    "two_hop": _two_hop,
    "agreement": _agreement,
    "comparison": _comparison,
    "weekday": _weekday,
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
