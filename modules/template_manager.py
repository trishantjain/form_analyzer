"""Persistent multi-template/profile manager for Form Analyzer."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_TEMPLATE = {
    "form_name": "New Form",
    "template_version": "2.0",
    "template_type": "generic",
    "reference_count": 0,
    "structure": {
        "anchors": [],
        "minimum_anchor_match": 0.70,
        "required_anchor_ratio": 0.70,
    },
    "visual_checks": [],
    "matching": {
        "enabled": True,
        "layout_threshold": 0.30,
        "warning_only": True,
    },
    "validation": {
        "ocr_match_threshold": 0.70,
        "require_anchor_ratio": 0.70,
        "require_all_mandatory": False,
    },
    "scoring": {
        "structure_weight": 70,
        "visual_weight": 30,
        "pass_score_threshold": 75,
    },
}


def slugify(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return value or "template"


class TemplateManager:
    def __init__(self, root: Path = Path("templates/profiles"), legacy_config: Path = Path("config/form_template.json")):
        self.root = root
        self.legacy_config = legacy_config
        self.root.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy()

    def _profile_dir(self, slug: str) -> Path:
        return self.root / slug

    def _config_path(self, slug: str) -> Path:
        return self._profile_dir(slug) / "template.json"

    def _references_dir(self, slug: str) -> Path:
        return self._profile_dir(slug) / "references"

    def _migrate_legacy(self) -> None:
        """Create a persistent CSR profile from the old single-template config once."""
        if any(self.root.iterdir()):
            return
        if self.legacy_config.exists():
            try:
                config = json.loads(self.legacy_config.read_text(encoding="utf-8"))
            except Exception:
                config = DEFAULT_TEMPLATE.copy()
            name = config.get("form_name") or "CSR"
        else:
            config = DEFAULT_TEMPLATE.copy()
            name = "CSR"
        self.create(name, config=config, overwrite=False)

    def list_templates(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            config_path = directory / "template.json"
            if not config_path.exists():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {"form_name": directory.name}
            refs = list((directory / "references").glob("*")) if (directory / "references").exists() else []
            items.append({
                "slug": directory.name,
                "name": config.get("form_name", directory.name),
                "reference_count": sum(1 for p in refs if p.is_file()),
                "path": str(directory),
            })
        return items

    def exists(self, name: str) -> bool:
        return self._profile_dir(slugify(name)).exists()

    def create(self, name: str, config: Optional[Dict[str, Any]] = None, overwrite: bool = False) -> str:
        slug = slugify(name)
        directory = self._profile_dir(slug)
        if directory.exists() and not overwrite:
            raise FileExistsError(f"Template '{name}' already exists")
        directory.mkdir(parents=True, exist_ok=True)
        self._references_dir(slug).mkdir(parents=True, exist_ok=True)
        data = json.loads(json.dumps(config if config is not None else DEFAULT_TEMPLATE))
        data["form_name"] = name
        self._config_path(slug).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return slug

    def duplicate(self, source_slug: str, new_name: str) -> str:
        source = self._profile_dir(source_slug)
        if not source.exists():
            raise FileNotFoundError(source_slug)
        new_slug = slugify(new_name)
        target = self._profile_dir(new_slug)
        if target.exists():
            raise FileExistsError(f"Template '{new_name}' already exists")
        shutil.copytree(source, target)
        config = self.load(new_slug)
        config["form_name"] = new_name
        self.save(new_slug, config)
        return new_slug

    def load(self, slug: str) -> Dict[str, Any]:
        path = self._config_path(slug)
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, slug: str, config: Dict[str, Any]) -> None:
        path = self._config_path(slug)
        if not path.exists():
            raise FileNotFoundError(path)
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_references(self, slug: str, files: List[Any]) -> int:
        target = self._references_dir(slug)
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        for uploaded in files:
            filename = Path(uploaded.name).name
            if not filename:
                continue
            destination = target / filename
            destination.write_bytes(uploaded.getvalue())
            count += 1
        return count

    def learn_from_references(self, slug: str) -> Dict[str, Any]:
        """Run OCR over saved reference images and rebuild the generic template profile."""
        from modules.template_analyzer import learn_template_profile

        refs = self.reference_files(slug)
        config = self.load(slug)
        form_name = config.get("form_name", slug)

        if not refs:
            raise ValueError("Add at least one reference image before learning the template.")

        learned = learn_template_profile(refs, form_name=form_name)
        self.save(slug, learned)
        return learned

    def reference_files(self, slug: str) -> List[Path]:
        directory = self._references_dir(slug)
        if not directory.exists():
            return []
        return sorted(p for p in directory.iterdir() if p.is_file())

    def delete(self, slug: str) -> None:
        directory = self._profile_dir(slug)
        if directory.exists():
            shutil.rmtree(directory)

    def rename(self, slug: str, new_name: str) -> str:
        new_slug = slugify(new_name)
        if new_slug == slug:
            config = self.load(slug)
            config["form_name"] = new_name
            self.save(slug, config)
            return slug
        source = self._profile_dir(slug)
        target = self._profile_dir(new_slug)
        if target.exists():
            raise FileExistsError(f"Template '{new_name}' already exists")
        source.rename(target)
        config = self.load(new_slug)
        config["form_name"] = new_name
        self.save(new_slug, config)
        return new_slug
