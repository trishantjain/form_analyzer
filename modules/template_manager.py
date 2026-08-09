"""Persistent generic template/profile manager."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# ONE AND ONLY TEMPLATE SCHEMA
# ============================================================

DEFAULT_TEMPLATE = {
    "form_name": "New Form",
    "template_version": "3.0",
    "template_type": "generic",
    "reference_count": 0,

    "page_geometry": {
        "aspect_ratio": None,
        "orientation": None,
        "aspect_ratio_tolerance": 0.12,
        "reference_sizes": [],
    },

    "structure": {
        "anchors": [],
        "minimum_anchor_match": 0.70,
        "required_anchor_ratio": 0.70,
    },

    "visual_checks": [],

    "matching": {
        "enabled": True,
        "layout_threshold": 0.30,
        "warning_only": False,
    },

    "validation": {
        "ocr_match_threshold": 0.70,
        "require_anchor_ratio": 0.70,
        "require_all_mandatory": True,
    },

    "scoring": {
        "structure_weight": 80,
        "visual_weight": 20,
        "pass_score_threshold": 75,
    },

    "metadata": {
        "learned_from": [],
        "reference_text_sample": [],
    },
}


def slugify(name: str) -> str:
    value = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        name.strip().lower(),
    ).strip("-")

    return value or "template"


def _deep_copy(data: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(data))


def _is_new_schema(config: Any) -> bool:
    if not isinstance(config, dict):
        return False

    return (
        isinstance(config.get("structure"), dict)
        and isinstance(config.get("visual_checks"), list)
        and isinstance(config.get("validation"), dict)
        and isinstance(config.get("scoring"), dict)
    )


def _convert_old_schema(
    old_config: Dict[str, Any],
    form_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert the old Form Analyzer configuration format
    into the new generic template format.
    """

    new_config = _deep_copy(DEFAULT_TEMPLATE)

    new_config["form_name"] = (
        form_name
        or old_config.get("form_name")
        or "New Form"
    )

    # --------------------------------------------------------
    # Old headings -> new anchors
    # --------------------------------------------------------

    anchors = []

    for index, heading in enumerate(
        old_config.get("required_headings", []) or []
    ):
        if isinstance(heading, str):
            label = heading
            aliases = [heading]
        elif isinstance(heading, dict):
            label = (
                heading.get("label")
                or heading.get("name")
                or heading.get("id")
                or f"Heading {index + 1}"
            )

            aliases = heading.get("aliases") or [label]

        else:
            continue

        anchors.append({
            "id": slugify(label),
            "label": label,
            "aliases": aliases,
            "reference_count": 1,
            "x": 0.5,
            "y": 0.5,
            "position_tolerance": 0.12,
        })

    new_config["structure"]["anchors"] = anchors

    # --------------------------------------------------------
    # Old signature configuration
    # --------------------------------------------------------

    old_elements = old_config.get(
        "required_elements",
        {},
    ) or {}

    signature = old_elements.get("signature", {}) or {}
    stamp = old_elements.get("stamp", {}) or {}

    visual_checks = []

    if signature.get("required"):
        visual_checks.append({
            "id": "signature",
            "label": "Signature",
            "type": "signature",
            "required": True,
            "region": {
                "x": 0.10,
                "y": 0.65,
                "width": 0.80,
                "height": 0.15,
            },
            "min_ink_percentage": 0.8,
        })

    if stamp.get("required"):
        visual_checks.append({
            "id": "stamp",
            "label": "Stamp / Seal",
            "type": "stamp",
            "required": True,
            "region": {
                "x": 0.10,
                "y": 0.65,
                "width": 0.80,
                "height": 0.15,
            },
            "min_ink_percentage": 0.5,
        })

    new_config["visual_checks"] = visual_checks

    # --------------------------------------------------------
    # Old scoring -> new scoring
    # --------------------------------------------------------

    old_scoring = old_config.get("scoring", {}) or {}

    new_config["scoring"]["pass_score_threshold"] = float(
        old_scoring.get(
            "pass_score_threshold",
            75,
        )
    )

    return new_config


def normalize_template_config(
    config: Optional[Dict[str, Any]],
    form_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Guarantee that every template stored by the application
    uses the current schema.
    """

    if not isinstance(config, dict):
        config = {}

    if not _is_new_schema(config):
        return _convert_old_schema(
            config,
            form_name=form_name,
        )

    result = _deep_copy(config)

    if form_name:
        result["form_name"] = form_name

    # Ensure newer fields exist even if an older new-schema
    # template is being loaded.
    defaults = _deep_copy(DEFAULT_TEMPLATE)

    if "page_geometry" not in result:
        result["page_geometry"] = defaults["page_geometry"]

    if "metadata" not in result:
        result["metadata"] = defaults["metadata"]

    result["template_version"] = "3.0"
    result["template_type"] = "generic"

    return result


class TemplateManager:

    def __init__(
        self,
        root: Path = Path("templates/profiles"),
        legacy_config: Path = Path("config/form_template.json"),
    ):
        self.root = root
        self.legacy_config = legacy_config

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._migrate_legacy()

    # ========================================================
    # PATHS
    # ========================================================

    def _profile_dir(self, slug: str) -> Path:
        return self.root / slug

    def _config_path(self, slug: str) -> Path:
        return (
            self._profile_dir(slug)
            / "template.json"
        )

    def _references_dir(self, slug: str) -> Path:
        return (
            self._profile_dir(slug)
            / "references"
        )

    # ========================================================
    # LEGACY MIGRATION
    # ========================================================

    def _migrate_legacy(self) -> None:

        if self.legacy_config.exists():

            try:
                config = json.loads(
                    self.legacy_config.read_text(
                        encoding="utf-8"
                    )
                )

                config = normalize_template_config(
                    config
                )

                name = (
                    config.get("form_name")
                    or "CSR"
                )

                if not self.exists(name):
                    self.create(
                        name,
                        config=config,
                        overwrite=False,
                    )

            except Exception:
                pass

    # ========================================================
    # LIST
    # ========================================================

    def list_templates(self) -> List[Dict[str, Any]]:

        items = []

        if not self.root.exists():
            return items

        for directory in sorted(
            self.root.iterdir()
        ):

            if not directory.is_dir():
                continue

            config_path = (
                directory
                / "template.json"
            )

            if not config_path.exists():
                continue

            try:

                config = json.loads(
                    config_path.read_text(
                        encoding="utf-8"
                    )
                )

                # Automatically repair old templates.
                normalized = normalize_template_config(
                    config,
                    form_name=config.get(
                        "form_name",
                        directory.name,
                    ),
                )

                if normalized != config:
                    config_path.write_text(
                        json.dumps(
                            normalized,
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

                    config = normalized

            except Exception:

                config = {
                    "form_name": directory.name
                }

            refs_dir = (
                directory / "references"
            )

            reference_count = 0

            if refs_dir.exists():
                reference_count = sum(
                    1
                    for p in refs_dir.iterdir()
                    if p.is_file()
                )

            items.append({
                "slug": directory.name,
                "name": config.get(
                    "form_name",
                    directory.name,
                ),
                "reference_count": reference_count,
                "path": str(directory),
            })

        return items

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(self, name: str) -> bool:

        return self._profile_dir(
            slugify(name)
        ).exists()

    # ========================================================
    # CREATE
    # ========================================================

    def create(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> str:

        slug = slugify(name)

        directory = self._profile_dir(slug)

        if (
            directory.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Template '{name}' already exists"
            )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._references_dir(
            slug
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        # THIS IS THE IMPORTANT PART.
        # Every newly created template is
        # forced into the new schema.
        data = normalize_template_config(
            config,
            form_name=name,
        )

        self._config_path(slug).write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return slug

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        slug: str,
    ) -> Dict[str, Any]:

        path = self._config_path(slug)

        if not path.exists():
            raise FileNotFoundError(path)

        config = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        # Automatically repair old configuration.
        normalized = normalize_template_config(
            config
        )

        if normalized != config:

            path.write_text(
                json.dumps(
                    normalized,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        return normalized

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        slug: str,
        config: Dict[str, Any],
    ) -> None:

        path = self._config_path(slug)

        if not path.exists():
            raise FileNotFoundError(path)

        config = normalize_template_config(
            config
        )

        path.write_text(
            json.dumps(
                config,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # REFERENCES
    # ========================================================

    def add_references(
        self,
        slug: str,
        files: List[Any],
    ) -> int:

        target = self._references_dir(
            slug
        )

        target.mkdir(
            parents=True,
            exist_ok=True,
        )

        count = 0

        for uploaded in files:

            filename = Path(
                uploaded.name
            ).name

            if not filename:
                continue

            destination = (
                target / filename
            )

            destination.write_bytes(
                uploaded.getvalue()
            )

            count += 1

        return count

    def reference_files(
        self,
        slug: str,
    ) -> List[Path]:

        directory = self._references_dir(
            slug
        )

        if not directory.exists():
            return []

        return sorted(
            p
            for p in directory.iterdir()
            if p.is_file()
        )

    # ========================================================
    # LEARN TEMPLATE
    # ========================================================

    def learn_from_references(
        self,
        slug: str,
    ) -> Dict[str, Any]:

        from modules.template_analyzer import (
            learn_template_profile
        )

        refs = self.reference_files(
            slug
        )

        if not refs:
            raise ValueError(
                "Add at least one reference image before learning the template."
            )

        config = self.load(slug)

        form_name = config.get(
            "form_name",
            slug,
        )

        learned = learn_template_profile(
            refs,
            form_name=form_name,
        )

        # Force the learned configuration
        # through the same schema normalizer.
        learned = normalize_template_config(
            learned,
            form_name=form_name,
        )

        learned["reference_count"] = len(refs)

        learned.setdefault(
            "metadata",
            {},
        )

        learned["metadata"][
            "learned_from"
        ] = [
            p.name for p in refs
        ]

        self.save(
            slug,
            learned,
        )

        return learned

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        slug: str,
    ) -> None:

        directory = self._profile_dir(
            slug
        )

        if directory.exists():
            shutil.rmtree(directory)

    # ========================================================
    # RENAME
    # ========================================================

    def rename(
        self,
        slug: str,
        new_name: str,
    ) -> str:

        new_slug = slugify(
            new_name
        )

        if new_slug == slug:

            config = self.load(slug)

            config["form_name"] = (
                new_name
            )

            self.save(
                slug,
                config,
            )

            return slug

        source = self._profile_dir(
            slug
        )

        target = self._profile_dir(
            new_slug
        )

        if target.exists():
            raise FileExistsError(
                f"Template '{new_name}' already exists"
            )

        source.rename(target)

        config = self.load(
            new_slug
        )

        config["form_name"] = (
            new_name
        )

        self.save(
            new_slug,
            config,
        )

        return new_slug

    # ========================================================
    # DUPLICATE
    # ========================================================

    def duplicate(
        self,
        source_slug: str,
        new_name: str,
    ) -> str:

        source = self._profile_dir(
            source_slug
        )

        if not source.exists():
            raise FileNotFoundError(
                source_slug
            )

        new_slug = slugify(
            new_name
        )

        target = self._profile_dir(
            new_slug
        )

        if target.exists():
            raise FileExistsError(
                f"Template '{new_name}' already exists"
            )

        shutil.copytree(
            source,
            target,
        )

        config = self.load(
            new_slug
        )

        config["form_name"] = (
            new_name
        )

        self.save(
            new_slug,
            config,
        )

        return new_slug