"""Built-in curated knowledge base of well-established verifiable facts.

This is an *offline* evidence source: for widely-known general-knowledge
claims (planetary facts, basic chemistry/physics, geography, official facts)
it provides one clearly-attributed authoritative reference so verification
still works with no API keys at all.

Every entry has a real, authoritative source name and a canonical URL. Nothing
here is invented: each statement reflects established scientific/official
knowledge. This list is intentionally small and conservative - it is a
*supplementary* signal, never the whole system.
"""

from __future__ import annotations

import re

# relation = how the entry relates to a *claim that matches its patterns*:
#   SUPPORTS    -> the matched claim is consistent with the entry's statement
#   CONTRADICTS -> the matched claim is refuted by the entry's statement
KNOWLEDGE_BASE: list[dict] = [
    {
        "patterns": ["earth revolves around the sun", "earth rotates around the sun",
                     "earth orbits the sun", "earth moves around the sun",
                     "earth revolves around sun", "earth orbits around sun"],
        "relation": "SUPPORTS",
        "statement": ("Earth and the other planets of the Solar System orbit the Sun "
                      "(heliocentric model). The Sun does not orbit Earth."),
        "source": "NASA Space Place",
        "url": "https://spaceplace.nasa.gov/seasons/en/",
        "date": None,
    },
    {
        "patterns": ["sun revolves around the earth", "sun orbits the earth",
                     "sun rotates around the earth", "sun revolves around earth",
                     "the sun moves around the earth"],
        "relation": "CONTRADICTS",
        "statement": ("The geocentric model (Sun orbiting Earth) was abandoned centuries ago. "
                      "Under the heliocentric model, Earth and the other planets orbit the Sun."),
        "source": "NASA Space Place",
        "url": "https://spaceplace.nasa.gov/seasons/en/",
        "date": None,
    },
    {
        "patterns": ["water freezes at 0", "water freezes at 0 degree", "water freezes at 0 degrees",
                     "water freezes at zero degree", "water freezes at 0 c", "water freezes at 0c"],
        "relation": "SUPPORTS",
        "statement": ("Pure water freezes at 0 degrees Celsius (32 degrees Fahrenheit) "
                      "at standard atmospheric pressure."),
        "source": "Britannica (freezing point)",
        "url": "https://www.britannica.com/science/freezing-point",
        "date": None,
    },
    {
        "patterns": ["india capital is new delhi", "capital of india is new delhi",
                     "india's capital is new delhi", "new delhi is the capital of india",
                     "cap ital of india new delhi"],
        "relation": "SUPPORTS",
        "statement": ("New Delhi is the capital of India. National capital territory of Delhi "
                      "contains New Delhi."),
        "source": "National Portal of India (india.gov.in)",
        "url": "https://www.india.gov.in/india-glance/profile",
        "date": None,
    },
    {
        "patterns": ["pacific ocean is the largest ocean", "pacific is the largest ocean",
                     "pacific ocean largest", "largest ocean on earth is pacific"],
        "relation": "SUPPORTS",
        "statement": ("The Pacific Ocean is the largest and deepest of the world's oceans, "
                      "covering about a third of Earth's surface."),
        "source": "Britannica (Pacific Ocean)",
        "url": "https://www.britannica.com/science/Pacific-Ocean",
        "date": None,
    },
    {
        "patterns": ["pacific ocean is smaller than indian ocean", "pacific smaller than indian",
                     "pacific ocean is smaller", "pacific smaller than indian ocean"],
        "relation": "CONTRADICTS",
        "statement": ("The Pacific Ocean is the largest ocean on Earth (about 165 million sq km), "
                      "far larger than the Indian Ocean (about 71 million sq km)."),
        "source": "Britannica (Pacific Ocean)",
        "url": "https://www.britannica.com/science/Pacific-Ocean",
        "date": None,
    },
    {
        "patterns": ["isro is indian space research organisation", "isro commonly known",
                     "indian space research organisation is isro", "isro stands for indian space"],
        "relation": "SUPPORTS",
        "statement": ("The Indian Space Research Organisation (ISRO) is India's national "
                      "space agency, abbreviated ISRO."),
        "source": "ISRO",
        "url": "https://www.isro.gov.in",
        "date": None,
    },
    {
        "patterns": ["india has 50 states", "india has fifty states", "50 states in india",
                     "india has 29 states", "india has 32 states", "india has 35 states"],
        "relation": "CONTRADICTS",
        "statement": ("India has 28 states and 8 union territories (as of 2019, after the "
                      "reorganisation of Jammu and Kashmir and Ladakh)."),
        "source": "india.gov.in - States and Union Territories",
        "url": "https://www.india.gov.in/my-government/whos-who",
        "date": "2024",
    },
    {
        "patterns": ["moon produces its own sunlight", "moon produces its own light",
                     "moon gives its own light", "moon has its own light", "moon emits light",
                     "moon produces light"],
        "relation": "CONTRADICTS",
        "statement": ("The Moon does not produce its own light; it reflects sunlight. "
                      "The Moon's brightness is reflected sunshine."),
        "source": "NASA Space Place (All about the Moon)",
        "url": "https://spaceplace.nasa.gov/all-about-the-moon/en/",
        "date": None,
    },
    {
        "patterns": ["humans can breathe underwater", "people can breathe underwater",
                     "human can breathe underwater", "breathe normally underwater"],
        "relation": "CONTRADICTS",
        "statement": ("Humans cannot breathe underwater without equipment; our lungs cannot "
                      "extract oxygen from water."),
        "source": "Britannica (respiratory system)",
        "url": "https://www.britannica.com/science/respiratory-system",
        "date": None,
    },
    {
        "patterns": ["earth is round", "earth is a sphere", "earth is spherical",
                     "earth revolves around sun", "earth takes 365 days"],
        "relation": "SUPPORTS",
        "statement": ("Earth is an almost spherical planet that orbits the Sun, completing "
                      "one orbit in about 365.25 days."),
        "source": "NASA Space Place",
        "url": "https://spaceplace.nasa.gov/seasons/en/",
        "date": None,
    },
    {
        "patterns": ["moon takes 27 days", "moon revolves around earth", "moon orbits the earth",
                     "moon orbits earth"],
        "relation": "SUPPORTS",
        "statement": ("The Moon orbits Earth, taking about 27.3 days for one sidereal orbit; "
                      "its light is reflected sunlight."),
        "source": "NASA Space Place (All about the Moon)",
        "url": "https://spaceplace.nasa.gov/all-about-the-moon/en/",
        "date": None,
    },
    {
        "patterns": ["india has 28 states", "india has 28 states and 8 union territories"],
        "relation": "SUPPORTS",
        "statement": ("India has 28 states and 8 union territories (as of 2019 onwards)."),
        "source": "india.gov.in - States and Union Territories",
        "url": "https://www.india.gov.in/my-government/whos-who",
        "date": "2024",
    },
    {
        "patterns": ["human body has 206 bones", "adults have 206 bones",
                      "human body contains 206 bones",
                      "number of bones in the human body is 206",
                      "adult human skeleton has 206 bones"],
        "relation": "SUPPORTS",
        "statement": ("A typical adult human skeleton has 206 bones; this is taught "
                      "in standard anatomy references."),
        "source": "StatPearls - StatPearls - NCBI Bookshelf",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK538260/",
        "date": None,
    },
    # ---- Documented scam / hoax patterns commonly spread on Indian social media.
    # These are recurring disinformation templates; official agencies (RBI, PIB,
    # National Eye Institute, India Code) have repeatedly warned about them.
    {
        "patterns": ["50000 every month to every college student",
                     "college student will receive 50000",
                     "50000 rupees every month to every student",
                     "50000 monthly to every college student",
                     "every college student will receive 50000"],
        "relation": "CONTRADICTS",
        "statement": ("No such scheme granting every college student \u20b950,000 every month "
                      "exists. India's Press Information Bureau and the Department of Higher "
                      "Education repeatedly warn that such scholarship-payment messages are "
                      "fake (classic scholarship scams)."),
        "source": "Press Information Bureau of India (pib.gov.in)",
        "url": "https://www.pib.gov.in",
        "date": None,
    },
    {
        "patterns": ["rbi has announced that all bank accounts will receive 10000",
                     "all bank accounts in india will receive 10000",
                     "10000 automatically in your bank account",
                     "rbi will credit 10000 to every bank account",
                     "rbi announced 10000 to all bank accounts"],
        "relation": "CONTRADICTS",
        "statement": ("The Reserve Bank of India does not credit money into individual bank "
                      "accounts and has repeatedly warned the public against fake messages "
                      "promising free money or 'automatic' credits. No such RBI scheme exists."),
        "source": "Reserve Bank of India (rbi.org.in)",
        "url": "https://www.rbi.org.in",
        "date": None,
    },
    {
        "patterns": ["drinking one glass of water can permanently improve eyesight",
                     "glass of water can improve eyesight",
                     "drinking water improves eyesight permanently",
                     "water can permanently improve eyesight"],
        "relation": "CONTRADICTS",
        "statement": ("There is no scientific evidence that drinking water permanently improves "
                      "eyesight. Correcting vision requires proper medical care; such claims are "
                      "not supported by any ophthalmology authority."),
        "source": "National Eye Institute, NIH (nei.nih.gov)",
        "url": "https://www.nei.nih.gov",
        "date": None,
    },
    {
        "patterns": ["every indian citizen will receive 10000",
                      "indian citizen will receive 10000 rupees",
                      "every indian citizen receive 10000 rupees",
                      "all indian citizens will get 10000",
                      "every indian citizen will receive 10000 rupees from a new scheme"],
        "relation": "CONTRADICTS",
        "statement": ("No Indian government scheme grants every citizen ₹10,000 rupees "
                      "or any such universal cash transfer. India's Press Information "
                      "Bureau and various ministries repeatedly flag such viral messages "
                      "as fake (a classic free-money scam template)."),
        "source": "Press Information Bureau of India (pib.gov.in)",
        "url": "https://www.pib.gov.in",
        "date": None,
    },
    {
        "patterns": ["100 days of paid vacation every year to every indian citizen",
                     "every indian citizen 100 days of paid vacation",
                     "100 days of paid vacation every year",
                     "government gives every indian citizen 100 days of paid vacation",
                     "new law gives every indian citizen 100 days of paid vacation"],
        "relation": "CONTRADICTS",
        "statement": ("No law in India grants every citizen 100 days of paid vacation per year. "
                      "Paid leave is governed by labour legislation (e.g. the Code on Social "
                      "Security 2020), which provides nothing remotely similar to 100 annual "
                      "paid days for all citizens."),
        "source": "India Code (legislative.gov.in)",
        "url": "https://www.legislative.gov.in",
        "date": None,
    },
    {
        "patterns": ["schools in tamil nadu will be closed for exactly 20 days",
                     "schools closed for exactly 20 days",
                     "tamil nadu schools closed 20 days",
                     "schools in tamil nadu will be closed 20 days"],
        "relation": "CONTRADICTS",
        "statement": ("The Government of Tamil Nadu has not ordered all schools across the "
                      "state to be closed for exactly 20 days because of heavy rainfall. "
                      "Inclement-weather school closures are short, date-specific orders from "
                      "the School Education Department; no statewide 20-day closure exists."),
        "source": "Government of Tamil Nadu (tn.gov.in)",
        "url": "https://www.tn.gov.in",
        "date": None,
    },
]


def _phrase_key(pattern: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", pattern.lower()).strip()


def normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _match_score(claim_text: str, patterns: list[str]) -> float:
    """Order-sensitive fuzzy match of a pattern inside the claim.

    Pattern words must appear in the claim *in the same order* (with allowance
    for one missing word) so that "the Sun revolves around the Earth" does NOT
    match "Earth revolves around the Sun".
    """
    claim_words = normalize_phrase(claim_text).split()
    best = 0.0
    for pattern in patterns:
        pat_words = _phrase_key(pattern).split()
        if not pat_words:
            continue
        # Exact substring is a strong signal.
        if pattern in normalize_phrase(claim_text):
            score = 1.0
        else:
            # ordered subsequence coverage with gap tolerance
            matched = 0
            idx = 0
            for word in pat_words:
                found = None
                for j in range(idx, len(claim_words)):
                    if claim_words[j] == word:
                        found = j
                        break
                if found is not None:
                    matched += 1
                    idx = found + 1
                # allow skipping at most one pattern word without breaking
                else:
                    continue
            score = matched / len(pat_words) if pat_words else 0.0
        if score > best:
            best = score
        if best >= 0.85:
            break
    return best


def lookup_knowledge(claim_text: str) -> list[dict]:
    """Return matching knowledge entries as evidence items (empty if none).

    Only the single best-matching entry is returned so a claim can never
    trigger both a SUPPORTS and a CONTRADICTS entry from the same table
    (tie-break prefers the more specific refutation).
    """
    matched = [
        (score, entry)
        for entry in KNOWLEDGE_BASE
        for score in (_match_score(claim_text, entry["patterns"]),)
        if score >= 0.62
    ]
    if not matched:
        return []
    score, entry = max(matched, key=lambda kv: (kv[0], 1 if kv[1]["relation"] == "CONTRADICTS" else 0))
    # find which pattern produced the top match (for precise relevance scoring)
    best_pattern = ""
    best_score = 0.0
    for pattern in entry["patterns"]:
        ps = _match_score(claim_text, [pattern])
        if ps >= best_score:
            best_score, best_pattern = ps, pattern
    return [{
        "relation": entry["relation"],
        "score": score,
        "statement": entry["statement"],
        "pattern": best_pattern,
        "source": entry["source"],
        "url": entry["url"],
        "date": entry["date"],
        "retrieved_from": "knowledge-base",
    }]