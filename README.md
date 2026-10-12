<!--

This source file is part of the Template Repository open-source project

SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Template Repository

[![REUSE status](https://api.reuse.software/badge/github.com/SchmiedmayerLab/TemplateRepository)](https://api.reuse.software/info/github.com/SchmiedmayerLab/TemplateRepository)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/SchmiedmayerLab/TemplateRepository/blob/main/LICENSE.md)

Starting point for a new Schmiedmayer Lab repository. It ships the organization baseline —
licensing, citation metadata, and the enforcement workflow — so a new project is compliant on its
first commit rather than retrofitted later.

**Work through [Setting up a new repository](#setting-up-a-new-repository) before writing code.** It
lists what to replace, which badges to add as the project grows, and the one place where ordering
matters.

## What is here

| Path | Purpose |
|---|---|
| `.github/workflows/repository-standards.yml` | calls the shared checker: REUSE, Markdown links, and the repository surface |
| `.github/dependabot.yml` | weekly GitHub Actions updates; add your language's ecosystem alongside |
| `.gitignore` | carries an SPDX header, like every other file here |
| `CITATION.cff` | citation metadata; Zenodo reads this at archive time |
| `CONTRIBUTORS.md` | project authors referenced by the SPDX headers |
| `LICENSE.md` | root licence, the only one GitHub and Zenodo detect |
| `LICENSES/MIT.txt` | licence text for REUSE |
| `REUSE.toml` | intentionally empty — see [As you add code](#3-as-you-add-code) |

Everything else — `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, issue and
pull request templates — is inherited from
[`SchmiedmayerLab/.github`](https://github.com/SchmiedmayerLab/.github) and must not be copied here.

## Setting up a new repository

Work through this top to bottom. Steps 1 and 2 take minutes; the rest happen as the project grows.
The order matters in exactly one place — Zenodo, in step 5 — where getting it wrong produces a
permanent, wrong DOI record.

### 1. Immediately after creating the repository

Replace `Template Repository` and `TemplateRepository` everywhere:

```bash
NAME="My Project"      # human-readable title
SLUG="MyProject"       # the GitHub repository name
YEAR=$(date +%Y)
grep -rl 'Template Repository\|TemplateRepository\|2026' . --exclude-dir=.git \
  | xargs sed -i '' -e "s/TemplateRepository/$SLUG/g" -e "s/Template Repository/$NAME/g" \
                    -e "s/SPDX-FileCopyrightText: 2026/SPDX-FileCopyrightText: $YEAR/g" \
                    -e "s/Copyright (c) 2026/Copyright (c) $YEAR/g"
```

Then:

- Add yourself to `CITATION.cff` and `CONTRIBUTORS.md`. The template ships one author because a
  template has one; a real project lists everyone who wrote it.
- Write the project description under the H1, above `## Contributing`.

Do **not** add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` or `SUPPORT.md`. They are
inherited from `SchmiedmayerLab/.github`, and a local copy will drift. The standards workflow warns
if it finds one.

### 2. Turn the checks on

`.github/workflows/repository-standards.yml` already runs on every pull request and on pushes to
`main`. After its first run, add **Check Repository Standards** to the repository's required status
checks — a check that only reports is a check people learn to ignore.

Verify locally before pushing:

```bash
reuse lint          # brew install reuse
actionlint          # brew install actionlint
```

### 3. As you add code

**Every file gets an SPDX header.** Copy the style already in `README.md`, adjusted to the
language's comment syntax:

```text
SPDX-FileCopyrightText: <year> Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
SPDX-License-Identifier: MIT
```

For files that cannot carry one — binaries, images, generated output, vendored dependencies — add
an annotation to `REUSE.toml` instead of leaving them uncovered.

Two things that are easy to get wrong:

- **Vendored third-party files keep their own licence.** A blanket MIT annotation over something
  like a Gradle wrapper is a false claim. Give it its real licence and add the text to `LICENSES/`.
- **Within `REUSE.toml` the last matching annotation wins.** Keep the blanket `path = ["**"]` entry
  first and put specific overrides below it, or the specific entry is silently ignored.

**Add a CI workflow** for the stack, then add its badge as the first entry in the block:

```markdown
[![Build and Test](https://github.com/SchmiedmayerLab/SLUG/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/SchmiedmayerLab/SLUG/actions/workflows/build-and-test.yml)
```

Badge order is fixed and enforced — **Build and Test · Deployment · CodeQL · Codecov · REUSE status ·
License: MIT · Release · DOI**. Only add a badge for something the repository actually has; the
checker fails a `CodeQL` badge with no `codeql.yml`, and vice versa.

### 4. Register the external services

> [!IMPORTANT]
> **Ask Paul Schmiedmayer to register these.** The account that registers a service owns the record
> it creates — the DOI, the coverage project, the REUSE entry. Registering from a personal account
> ties the repository to whoever happened to set it up.

- **REUSE** — <https://api.reuse.software/register>. Until this is done the badge reads
  `unregistered`, which looks like a failure. Confirm with
  `curl -s https://api.reuse.software/badge/github.com/SchmiedmayerLab/SLUG | grep unregistered`
  returning nothing.
- **Codecov** — activate at <https://app.codecov.io/gh/SchmiedmayerLab>. A public repository
  uploads without a token, so enable coverage reporting wherever there is a test suite.
- **Zenodo** — enable the repository at <https://zenodo.org/account/settings/github/>. Do this
  **before** the first release.

### 5. The first release

The order here is the part worth reading twice.

1. **Enable Zenodo first.** Only releases published *after* the integration is switched on are
   archived. Existing tags are never picked up retroactively.
2. **Land `CITATION.cff` before tagging.** Zenodo reads it from the tarball *at the tag*. Metadata
   merged afterwards never appears in that DOI record, and the record cannot be edited into
   correctness later.
3. **Publish the release.**
4. **Copy the concept DOI** from the Zenodo GitHub settings page into both the README badge and
   `CITATION.cff` as `doi:`. Use the **concept DOI** — stable across every version — not the
   per-version DOI, so the badge keeps tracking the newest release without edits.

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.CONCEPT_ID.svg)](https://doi.org/10.5281/zenodo.CONCEPT_ID)
```

Never use the legacy `zenodo.org/badge/<github-repo-id>.svg` form. It keeps rendering a
plausible-looking badge when copied into another repository, which is how two repositories in this
organization spent years advertising a DOI belonging to an unrelated project. The checker rejects
it.

The checker warns once a repository has releases but no DOI badge, so this step is hard to forget.

### 6. Ongoing

The standards workflow fails a pull request when a required file disappears, `CITATION.cff` stops
parsing or its `url` stops matching the repository, a badge label drifts, badges fall out of order,
a badge points at another repository, or the DOI in the README and in `CITATION.cff` disagree.

> [!IMPORTANT]
> **Everything below this line is fixed text and is identical in every repository** — the
> Contributing, License, Citation and Our Research sections, and the footer images. Replace
> `Template Repository` with the project name and leave the rest exactly as it is. The standards
> workflow checks these sections, so an edit here fails the build; if the wording needs to change,
> change it in `SchmiedmayerLab/.github` and it changes everywhere at once.

## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/SchmiedmayerLab/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/SchmiedmayerLab/.github/blob/main/CODE_OF_CONDUCT.md) first. You can find a list of contributors in the [CONTRIBUTORS.md](CONTRIBUTORS.md) file.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for more information.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff), which GitHub surfaces through the [*Cite this repository*](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) button.

## Our Research

For more information, visit the [Schmiedmayer Lab GitHub organization](https://github.com/SchmiedmayerLab).

![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-light.png#gh-light-mode-only)
![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-dark.png#gh-dark-mode-only)
