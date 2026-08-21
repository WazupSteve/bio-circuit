import random
from itertools import product
from typing import Any

from circuit_supernodes.models import FrozenStrictModel


class CircuitSample(FrozenStrictModel):
    family: str
    prompt: str
    target: str
    possible_answers: tuple[str, ...]
    source_family: str | None = None
    goal: str | None = None

    def __init__(
        self,
        family: str,
        prompt: str,
        target: str,
        possible_answers: tuple[str, ...],
        source_family: str | None = None,
        goal: str | None = None,
        **data: Any,
    ) -> None:
        super().__init__(
            **{
                "family": family,
                "prompt": prompt,
                "target": target,
                "possible_answers": possible_answers,
                "source_family": source_family,
                "goal": goal,
                **data,
            }
        )

    def to_row(self) -> dict:
        row = self.model_dump()
        # A cohort goal is the semantic mechanism prompts are meant to share.
        # Most families have one mechanism; generators with meaningful
        # subfamilies set ``goal`` explicitly below.
        row["goal"] = self.goal or self.source_family or self.family
        return row


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
TWO_HOP_ANSWERS = tuple(dict.fromkeys(row[2] for row in TWO_HOP))

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
ANTONYM_ANSWERS = {
    language: tuple(dict.fromkeys(answer for _, answer in pairs))
    for language, pairs in ANTONYMS.items()
}
ANALOGY_ANSWERS = tuple(dict.fromkeys(row[3] for row in ANALOGIES))
ACRONYM_ANSWERS = tuple(
    "".join(initials)
    for initials in product(
        sorted({word[0] for word in ACRONYM_WORDS["adj2"]}),
        sorted({word[0] for word in ACRONYM_WORDS["noun"]}),
        sorted({word[0] for word in ACRONYM_WORDS["suffix"]}),
    )
)
SEQUENCE_ANSWERS = tuple(
    dict.fromkeys(
        str(step * factor)
        for step in (3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17)
        for factor in range(5, 16)
    )
)
FACTUAL_ANSWERS = tuple(dict.fromkeys(sport for _, sport in FACTUAL_SPORTS))
INCONTEXT_ANSWERS = tuple(dict.fromkeys(value for _, value in INCONTEXT_ATTRS))


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
        possible_answers=TWO_HOP_ANSWERS,
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
        possible_answers=ANTONYM_ANSWERS[lang],
        goal=f"antonym:{lang}",
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
        possible_answers=ANALOGY_ANSWERS,
        goal=f"analogy:{rel}",
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
        possible_answers=ACRONYM_ANSWERS,
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
        possible_answers=SEQUENCE_ANSWERS,
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
        possible_answers=FACTUAL_ANSWERS,
    )


def _incontext(rng: random.Random) -> CircuitSample:
    pairs = rng.sample(INCONTEXT_ATTRS, 5)
    ctx, (qk, qv) = pairs[:4], pairs[4]
    prompt = " ".join(f"{k}: {v}" for k, v in ctx) + f" {qk}:"
    return CircuitSample(
        family="incontext",
        prompt=prompt,
        target=f" {qv}",
        possible_answers=INCONTEXT_ANSWERS,
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
        possible_answers=("is", "are"),
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
        possible_answers=("Yes", "No"),
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
        possible_answers=tuple(WEEKDAYS),
    )


SYMBOLS = (
    "Aster",
    "Birch",
    "Cobalt",
    "Dune",
    "Ember",
    "Fjord",
    "Grove",
    "Harbor",
    "Ivory",
    "Jasper",
    "Kestrel",
    "Lumen",
    "Mica",
    "Nimbus",
    "Onyx",
    "Pine",
    "Quartz",
    "Reed",
    "Sable",
    "Tundra",
)

PROPERTIES = (
    "daxen",
    "fepal",
    "glim",
    "koral",
    "loric",
    "marn",
    "nubic",
    "prax",
    "quell",
    "riven",
    "sovel",
    "tarsic",
)


def _path_composition(rng: random.Random) -> CircuitSample:
    """Follow a shuffled three-edge path while ignoring a disconnected edge."""

    start, first, second, answer, distractor_a, distractor_b = rng.sample(SYMBOLS, 6)
    facts = [
        f"{start} -> {first}",
        f"{first} -> {second}",
        f"{second} -> {answer}",
        f"{distractor_a} -> {distractor_b}",
    ]
    rng.shuffle(facts)
    return CircuitSample(
        family="path_composition",
        prompt=(
            "Follow the directed route facts exactly.\n"
            + "\n".join(facts)
            + f"\nStarting at {start} and following exactly three arrows reaches"
        ),
        target=f" {answer}",
        possible_answers=SYMBOLS,
    )


def _modular_arithmetic(rng: random.Random) -> CircuitSample:
    """Compose multiplication, signed addition, and a modular reduction."""

    def problem() -> tuple[int, int, int, int, int]:
        start = rng.randint(7, 39)
        multiplier = rng.randint(2, 6)
        delta = rng.choice([value for value in range(-17, 18) if value])
        modulus = rng.choice((7, 9, 11, 13, 16, 17))
        answer = (start * multiplier + delta) % modulus
        return start, multiplier, delta, modulus, answer

    exemplar = problem()
    query = problem()
    while query == exemplar:
        query = problem()

    def render(values: tuple[int, int, int, int, int], include_answer: bool) -> str:
        start, multiplier, delta, modulus, answer = values
        signed_delta = f"+ {delta}" if delta > 0 else f"- {abs(delta)}"
        suffix = str(answer) if include_answer else ""
        return f"(({start} * {multiplier}) {signed_delta}) mod {modulus} = {suffix}"

    return CircuitSample(
        family="modular_arithmetic",
        prompt=render(exemplar, True) + "\n" + render(query, False),
        target=str(query[-1]),
        possible_answers=tuple(str(value) for value in range(17)),
    )


def _rule_composition(rng: random.Random) -> CircuitSample:
    """Apply a shuffled three-rule implication chain with a distractor rule."""

    p0, p1, p2, answer, d0, d1 = rng.sample(PROPERTIES, 6)
    entity = rng.choice(SYMBOLS)
    rules = [
        f"Every {p0} thing is {p1}.",
        f"Every {p1} thing is {p2}.",
        f"Every {p2} thing is {answer}.",
        f"Every {d0} thing is {d1}.",
    ]
    rng.shuffle(rules)
    return CircuitSample(
        family="rule_composition",
        prompt=(" ".join(rules) + f" {entity} is {p0}. Therefore, {entity} is"),
        target=f" {answer}",
        possible_answers=PROPERTIES,
    )


DIRECTIONS = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
}


def _spatial_composition(rng: random.Random) -> CircuitSample:
    """Add four signed displacement vectors and classify the final direction."""

    while True:
        moves = [(rng.choice(tuple(DIRECTIONS)), rng.randint(1, 5)) for _ in range(4)]
        x = sum(DIRECTIONS[direction][0] * distance for direction, distance in moves)
        y = sum(DIRECTIONS[direction][1] * distance for direction, distance in moves)
        if x or y:
            break
    vertical = "north" if y > 0 else "south" if y < 0 else ""
    horizontal = "east" if x > 0 else "west" if x < 0 else ""
    answer = vertical + horizontal
    steps = ", then ".join(f"{direction} {distance}" for direction, distance in moves)
    traveler = rng.choice(SYMBOLS)
    return CircuitSample(
        family="spatial_composition",
        prompt=(
            "Answer with north, south, east, west, northeast, northwest, "
            "southeast, or southwest. "
            f"{traveler} moves {steps}. Relative to the starting point, "
            f"{traveler} is"
        ),
        target=f" {answer}",
        possible_answers=(
            "north",
            "south",
            "east",
            "west",
            "northeast",
            "northwest",
            "southeast",
            "southwest",
        ),
    )


def _boolean_composition(rng: random.Random) -> CircuitSample:
    """Evaluate a balanced three-gate Boolean circuit."""

    a, b, c = (rng.random() < 0.5 for _ in range(3))
    first_gate = rng.choice(("AND", "OR"))
    second_gate = "OR" if first_gate == "AND" else "AND"
    x = a and b if first_gate == "AND" else a or b
    y = x or c if second_gate == "OR" else x and c
    invert = rng.random() < 0.5
    answer = not y if invert else y
    final = "NOT Y" if invert else "Y"

    def truth(value: bool) -> str:
        return "true" if value else "false"

    return CircuitSample(
        family="boolean_composition",
        prompt=(
            f"A={truth(a)}, B={truth(b)}, C={truth(c)}. "
            f"X = A {first_gate} B. Y = X {second_gate} C. "
            f"Z = {final}. Z is"
        ),
        target=f" {truth(answer)}",
        possible_answers=("true", "false"),
    )


TRAIN_FAMILIES = (
    "antonym",
    "analogy",
    "acronym",
    "sequence",
    "factual",
    "incontext",
)

EVAL_FAMILIES = (
    "two_hop",
    "agreement",
    "comparison",
    "weekday",
    "path_composition",
    "modular_arithmetic",
    "rule_composition",
    "spatial_composition",
    "boolean_composition",
)

SEALED_FAMILIES = EVAL_FAMILIES

FAMILIES = {
    "two_hop": _two_hop,
    "agreement": _agreement,
    "comparison": _comparison,
    "weekday": _weekday,
    "path_composition": _path_composition,
    "modular_arithmetic": _modular_arithmetic,
    "rule_composition": _rule_composition,
    "spatial_composition": _spatial_composition,
    "boolean_composition": _boolean_composition,
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
    # An unqualified build is training-only. Sealed evaluation families must be
    # requested explicitly, which prevents accidental train/eval contamination.
    fams = list(TRAIN_FAMILIES) if families is None else list(families)
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


def generate_pool(n: int, seed: int, families: list[str]) -> list[CircuitSample]:
    """Generate an unbalanced cohort pool without one small family capping all.

    Cohort builders need hundreds of candidates so answers repeat. Some finite
    families have fewer unique prompts than others; exhausted families are
    retired while the remaining requested families continue sampling.
    """

    if n < 1:
        raise ValueError("n must be >= 1")
    if not families:
        raise ValueError("families must not be empty")
    if unknown := set(families) - FAMILIES.keys():
        raise ValueError(f"unknown families: {sorted(unknown)}")
    rng = random.Random(seed)
    active = list(dict.fromkeys(families))
    misses = {family: 0 for family in active}
    seen: set[str] = set()
    samples = []
    while active and len(samples) < n:
        family = rng.choice(active)
        sample = FAMILIES[family](rng)
        if sample.prompt in seen:
            misses[family] += 1
            if misses[family] >= 500:
                active.remove(family)
            continue
        misses[family] = 0
        seen.add(sample.prompt)
        samples.append(sample)
    if len(samples) < n:
        raise ValueError(
            f"requested families produced only {len(samples)} unique prompts"
        )
    rng.shuffle(samples)
    return samples
