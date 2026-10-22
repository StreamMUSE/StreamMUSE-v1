"""Reviewed character catalogs for the Character Guessing game."""

from __future__ import annotations

import re
from dataclasses import dataclass


_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z .'-]{0,79}\Z")
_ID_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")


def canonicalize_character_name(value: object) -> str:
    text = str(value or "").strip().lower().rstrip("?.!,;:")
    text = re.sub(r"[.'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class CharacterCandidate:
    character_id: str
    display_name: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.character_id):
            raise ValueError(f"invalid character id: {self.character_id!r}")
        for name in (self.display_name, *self.aliases):
            if not _NAME_PATTERN.fullmatch(name):
                raise ValueError(f"invalid character name or alias: {name!r}")


@dataclass(frozen=True)
class CharacterScope:
    scope_id: str
    title: str
    language: str
    candidates: tuple[CharacterCandidate, ...]

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.scope_id):
            raise ValueError(f"invalid scope id: {self.scope_id!r}")
        if not self.title.strip():
            raise ValueError("scope title must not be empty")
        if self.language != "en":
            raise ValueError("V1 character scopes must use English")
        if not 2 <= len(self.candidates) <= 100:
            raise ValueError("a character scope must contain between 2 and 100 candidates")

        ids: set[str] = set()
        names: dict[str, str] = {}
        for candidate in self.candidates:
            if candidate.character_id in ids:
                raise ValueError(f"duplicate character id: {candidate.character_id}")
            ids.add(candidate.character_id)
            for name in (candidate.display_name, *candidate.aliases):
                canonical = canonicalize_character_name(name)
                existing = names.get(canonical)
                if existing is not None and existing != candidate.character_id:
                    raise ValueError(
                        f"character name collision: {name!r} maps to {existing} and "
                        f"{candidate.character_id}"
                    )
                names[canonical] = candidate.character_id

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.character_id for candidate in self.candidates)

    def resolve(self, value: object) -> CharacterCandidate | None:
        canonical = canonicalize_character_name(value)
        for candidate in self.candidates:
            names = (candidate.display_name, *candidate.aliases)
            if canonical in {canonicalize_character_name(name) for name in names}:
                return candidate
        return None


def _candidate(character_id: str, display_name: str, *aliases: str) -> CharacterCandidate:
    return CharacterCandidate(character_id, display_name, tuple(aliases))


WORLD_HISTORICAL_FIGURES_EN = CharacterScope(
    scope_id="world_historical_figures_en",
    title="World Historical Figures",
    language="en",
    candidates=tuple(
        sorted(
            (
                _candidate("abraham_lincoln", "Abraham Lincoln"),
                _candidate("ada_lovelace", "Ada Lovelace"),
                _candidate("albert_einstein", "Albert Einstein", "Einstein"),
                _candidate("alexander_the_great", "Alexander the Great"),
                _candidate("amelia_earhart", "Amelia Earhart"),
                _candidate("aristotle", "Aristotle"),
                _candidate("charles_darwin", "Charles Darwin", "Darwin"),
                _candidate("cleopatra", "Cleopatra"),
                _candidate("confucius", "Confucius"),
                _candidate("florence_nightingale", "Florence Nightingale"),
                _candidate("frida_kahlo", "Frida Kahlo"),
                _candidate("galileo_galilei", "Galileo Galilei", "Galileo"),
                _candidate("genghis_khan", "Genghis Khan"),
                _candidate("george_washington", "George Washington"),
                _candidate("harriet_tubman", "Harriet Tubman"),
                _candidate("isaac_newton", "Isaac Newton", "Newton"),
                _candidate("jane_austen", "Jane Austen"),
                _candidate("joan_of_arc", "Joan of Arc"),
                _candidate("johannes_gutenberg", "Johannes Gutenberg", "Gutenberg"),
                _candidate("julius_caesar", "Julius Caesar", "Caesar"),
                _candidate("leonardo_da_vinci", "Leonardo da Vinci", "Da Vinci"),
                _candidate("louis_pasteur", "Louis Pasteur", "Pasteur"),
                _candidate("mahatma_gandhi", "Mahatma Gandhi", "Gandhi"),
                _candidate("marco_polo", "Marco Polo"),
                _candidate("marie_curie", "Marie Curie", "Curie"),
                _candidate(
                    "martin_luther_king_jr",
                    "Martin Luther King Jr.",
                    "Martin Luther King",
                    "MLK",
                ),
                _candidate("michelangelo", "Michelangelo"),
                _candidate("muhammad_ali", "Muhammad Ali"),
                _candidate("napoleon_bonaparte", "Napoleon Bonaparte", "Napoleon"),
                _candidate("nelson_mandela", "Nelson Mandela", "Mandela"),
                _candidate("nikola_tesla", "Nikola Tesla", "Tesla"),
                _candidate("pablo_picasso", "Pablo Picasso", "Picasso"),
                _candidate("rosa_parks", "Rosa Parks"),
                _candidate("sacagawea", "Sacagawea"),
                _candidate("sigmund_freud", "Sigmund Freud", "Freud"),
                _candidate("socrates", "Socrates"),
                _candidate("stephen_hawking", "Stephen Hawking", "Hawking"),
                _candidate("susan_b_anthony", "Susan B. Anthony", "Susan Anthony"),
                _candidate("thomas_edison", "Thomas Edison", "Edison"),
                _candidate("vincent_van_gogh", "Vincent van Gogh", "Van Gogh"),
                _candidate("william_shakespeare", "William Shakespeare", "Shakespeare"),
                _candidate("winston_churchill", "Winston Churchill", "Churchill"),
                _candidate(
                    "wolfgang_amadeus_mozart",
                    "Wolfgang Amadeus Mozart",
                    "Mozart",
                ),
                _candidate("yuri_gagarin", "Yuri Gagarin", "Gagarin"),
            ),
            key=lambda candidate: candidate.display_name,
        )
    ),
)


CHARACTER_SCOPES: dict[str, CharacterScope] = {
    WORLD_HISTORICAL_FIGURES_EN.scope_id: WORLD_HISTORICAL_FIGURES_EN,
}


def get_character_scope(scope_id: str) -> CharacterScope:
    try:
        return CHARACTER_SCOPES[str(scope_id)]
    except KeyError as exc:
        choices = ", ".join(sorted(CHARACTER_SCOPES))
        raise ValueError(
            f"unknown character scope {scope_id!r}; choose one of: {choices}"
        ) from exc
