# AI Accessibility Analysis & Cleanup Candidates

**Date:** 2026-07-22  
**Change:** `improve-documentation`  
**Focus:** AI accessibility + code cleanup

## Summary

Analyzed vdf-ml repository for AI accessibility. Created 5 AI guide files in `docs/ai-guides/`. Identified cleanup candidates to reduce AI confusion.

## AI Guides Created

In `docs/ai-guides/`:
1. `AI-AGENT-GUIDE.md` - LLM-optimized overview
2. `DECISION-ROUTING.md` - Physics vs ML classification
3. `KEY-CONCEPTS.md` - Essential terms
4. `AI-TASK-TEMPLATES.md` - Common workflows
5. `AI-PITFALLS.md` - Common mistakes

## Cleanup Candidates (Priority Order)

**Correction (post-review)**: an earlier version of this document ranked
"Data Format Bridge" as candidate #1. That was a misdiagnosis -- the new
snapshot pipeline (`X.npy + metadata.csv`) and the old train/predict tools
(`X.npy/y.npy/metadata.csv`) are two independent, permanently separate
pipelines by design (see README's "known gap" and `docs/PCA_GUIDE.md`),
not an accidental incompatibility awaiting a bridge. It's removed from
this list; don't propose building one unless the user explicitly asks.

### 1. Deprecated Code Verification
**Check**: `src/deprecated.py` usage
**Verify**: Nothing depends on deprecated functions
**Files**: `src/deprecated.py`, `docs/DEPRECATED.md`

### 2. Batch Iteration Redundancy
**Check**: New redundant batch implementations
**Files**: `src/data_proc/batches.py`

### 3. Configuration Consistency
**Check**: All physics scripts use `pipeline_config.py`
**Files**: `scripts/data_proc/*.py`

### 4. Cross-Import Enforcement
**Rule**: `data_proc` never imports `ml_models`
**Check**: `grep -rn "from src.ml_models" src/data_proc/`
**Files**: `src/data_proc/`

## Next Step

Start cleanup with candidate #1 (deprecated code verification) - lowest
risk, confirms nothing else depends on `src/deprecated.py` before touching
anything else on this list.

## Current AI Accessibility Assessment

### Strengths for AI
1. **Clear file organization**: Physics vs ML separation in `src/data_proc/physics/` vs `src/ml_models/`
2. **README clarity**: "cluster_phys vs cluster_ml" distinction prevents common confusion
3. **Function indices**: PHYSICS.md, DATA_PROC.md, ML_MODELS.md provide file mapping
4. **Explicit warnings**: README documents common cross-domain traps

### Critical AI Accessibility Gaps

#### 1. **No AI-Specific Onboarding**
- No guide for AI agents on how to navigate this repository
- No decision tree for "Is this physics or ML?" routing
- No context optimization for LLM token limits

#### 2. **Implicit Architectural Knowledge**
- Why two PCA pipelines exist (snapshot clustering vs legacy CNN-dataset)
- Substance taxonomy and region classification rationale
- Feature representation tradeoffs (raw vs Hermite)
- Historical evolution (removed features: boundary_layer, margin_di)

#### 3. **Missing AI Metadata**
- No LLM-friendly summaries of key files
- No structured decision prompts for common tasks
- No optimization for context window limitations

#### 4. **Complex Decision Routing**
- Physics detection vs ML analysis boundaries unclear
- Multiple valid approaches with different tradeoffs
- Cross-domain coordination patterns undocumented

## AI Agent Needs Analysis

### **Task Comprehension**
- **Need**: Understand whether a task is physics, ML, or both
- **Current gap**: Agents must infer from code structure
- **Impact**: Wrong skill loading, wasted context, incorrect assumptions

### **Context Optimization** 
- **Need**: Minimal context to understand repository structure
- **Current gap**: Must read multiple files to grasp architecture
- **Impact**: Token waste, incomplete understanding

### **Decision Support**
- **Need**: Clear guidance on approach selection
- **Current gap**: Implicit knowledge in code/comments
- **Impact**: Suboptimal implementation choices

### **Error Prevention**
- **Need**: Awareness of common pitfalls
- **Current gap**: Some documented in README, not structured for AI
- **Impact**: Repeated mistakes, debugging overhead

## AI Accessibility Improvement Approaches

### Level 1: Minimal AI Onboarding (1-2 days)
1. **AI Agent Guide**: `AI-AGENT-GUIDE.md` with repository overview
2. **Decision Flowchart**: Visual/text decision tree for task routing
3. **Key Concepts Cheat Sheet**: Must-know terms and relationships
4. **File Purpose Summaries**: LLM-friendly descriptions of key files

### Level 2: Structured AI Metadata (3-4 days)
1. **Complete Level 1** plus:
2. **AI-focused README sections**: Structured for agent consumption
3. **Task Templates**: Common operation patterns with examples
4. **Context Optimization Guides**: What to read for different task types
5. **Skill Loading Guidance**: When to use which domain knowledge

### Level 3: AI-Optimized Repository (1 week)
1. **Complete Level 2** plus:
2. **LLM-specific documentation files**: `.agent-context.md` in key directories
3. **Automated context generation**: Scripts to create agent-friendly summaries
4. **Decision support prompts**: Ready-to-use prompts for common tasks
5. **Testing with AI agents**: Validation of AI comprehension

## Scope Estimation

### Immediate Priority (AI Accessibility Focus)
1. **AI Agent Guide** (0.5 day): `AI-AGENT-GUIDE.md`
2. **Decision Routing Chart** (0.5 day): Visual/text decision tree
3. **Key Files Metadata** (1 day): LLM-friendly summaries of 20 key files
4. **Task Templates** (1 day): 5-7 common operation patterns

### Total: 3 days (not 3 weeks)

## Risks

### Technical Risks
- **Over-engineering**: Creating unnecessary AI-specific infrastructure
- **Maintenance burden**: Additional files to update
- **Agent diversity**: Different AI agents have different needs

### Mitigation Strategies
1. **Minimal viable approach**: Start with simple guide and metadata
2. **Integration with existing docs**: Augment, don't replace
3. **Focus on high-impact areas**: Physics/ML boundary, decision routing
4. **Test with actual agents**: Validate usefulness with LLM workflows

## Skill Resolution
- **Physics domain knowledge**: Covered by load_physicist_knowledge skill
- **ML domain knowledge**: Covered by load_ml_knowledge skill  
- **Repository structure knowledge**: Covered by orchestrate skill
- **AI/documentation skills**: cognitive-doc-design skill applicable

## New Observations from Codebase Analysis

### AI Accessibility Challenges Found

1. **Domain Ambiguity Critical**: Repository structure creates natural confusion for AI agents
   - Physics (`data_proc/physics/`) vs ML (`ml_models/`) separation
   - Cross-domain tasks require coordination but layers separated
   - Skill system (`orchestrate`) helps but needs clear documentation

2. **Terminology Duplication**: `cluster_phys` vs `cluster_ml` is major confusion point
   - Different purposes, different computations
   - AI agents frequently conflate them
   - Requires explicit distinction in AI documentation

3. **Two Dataset Formats**: intentional, not a limitation
   - New extraction: `X.npy` + `metadata.csv` (labels in metadata)
   - Old tools: Expect `X.npy`/`y.npy`/`metadata.csv` triple
   - Two independent, permanently separate pipelines by design -- training
     uses the old format via `src/deprecated.py`'s `create_dataset`, not a
     conversion of the new pipeline's output

4. **Configuration Complexity**: Multiple config systems
   - Physics: Shared `pipeline_config.py` (Python)
   - ML: `configs/*.yaml` files
   - Different philosophies, need clear guidance for AI

5. **Architectural Implicit Knowledge**: Key decisions not obvious to AI
   - Why two PCA pipelines (snapshot vs dataset)
   - Substance taxonomy rationale
   - Layer separation (`data_proc` ≠ `ml_models`) reasoning

### Cleanup Candidates Identified

#### High Priority (Affects AI Workflow)

Not a candidate: "Data Format Bridge Implementation" was listed here in an
earlier version of this document. It's a misdiagnosis -- the new extraction
format and the old training tools are two independent, permanently
separate pipelines by design (see the correction above), not an
incompatibility to fix. AI agents doing end-to-end work should use the
snapshot pipeline's own end-to-end path (extraction → labelling → snapshot
PCA), not assume training is part of it.

1. **Deprecated Code Verification**
   - **Check**: `src/deprecated.py` actual usage
   - **Verify**: Nothing still depends on deprecated functions
   - **Action**: Remove if truly unused, or document active usage
   - **Files**: `src/deprecated.py`, `docs/DEPRECATED.md`

2. **Batch Iteration Redundancy Check**
   - **History**: `iter_index_batches` consolidated from 4 implementations
   - **Check**: New redundant implementations since consolidation
   - **Action**: Consolidate to `batches.py` if found
   - **Files**: `src/data_proc/batches.py`, grep for batch patterns

#### Medium Priority (Code Quality)

4. **Configuration Consistency Audit**
   - **Check**: All physics scripts use `pipeline_config.py`
   - **Verify**: No script-specific parameter blocks
   - **Action**: Consolidate stray configurations
   - **Files**: All `scripts/data_proc/*.py`, `scripts/ml_models/*.py`

5. **Cross-Import Rule Enforcement**
   - **Rule**: `data_proc` never imports `ml_models`
   - **Check**: `grep -rn "from src.ml_models" src/data_proc/`
   - **Action**: Fix any violations found
   - **Files**: `src/data_proc/` directory

6. **PCA Pipeline Documentation**
   - **Current**: Snapshot PCA (production) vs Dataset PCA (legacy)
   - **Clarify**: Purpose, usage, deprecation status
   - **Action**: Clear documentation for AI agents
   - **Files**: PCA-related scripts, documentation

#### Low Priority (Documentation)

7. **Substance Taxonomy AI Optimization**
   - **Current**: Well-documented in `SCHEMA.md`
   - **AI need**: Quick-reference for agent comprehension
   - **Action**: Added to AI artifacts (done)
   - **Files**: `docs/AI-AGENT-GUIDE.md`, `docs/KEY-CONCEPTS.md`

### AI Artifacts Created

Based on analysis, created 5 core AI accessibility files in `docs/`:

1. **`AI-AGENT-GUIDE.md`** - LLM-optimized repository overview
2. **`DECISION-ROUTING.md`** - Physics vs ML task classification  
3. **`KEY-CONCEPTS.md`** - Essential terms and relationships
4. **`AI-TASK-TEMPLATES.md`** - Common operation patterns
5. **`AI-PITFALLS.md`** - Common mistakes and prevention

**Design principles**:
- Token-efficient for LLM context windows
- Clear decision routing for domain ambiguity
- Prevention of common AI mistakes
- Integration with existing skill system

## Recommendations

### Immediate Next Step: Code Cleanup
Based on observations, prioritize cleanup in this order (data format
bridge removed from this list -- see correction above, it's not a real
gap):

1. **Deprecated code verification** - Remove dead weight
2. **Configuration consistency** - Prevent AI confusion
3. **Cross-import enforcement** - Maintain architectural integrity

### AI Accessibility Success Metrics
- AI agents correctly route tasks (physics/ML/both) 90%+ of time
- Core concepts understandable within 1K tokens
- Fewer repeated mistakes on common pitfalls
- AI comprehension of key architectural decisions

## Artifact Storage
- Exploration saved to: `openspec/changes/improve-documentation/explore.md`
- Topic key for memory: `sdd/improve-documentation/explore`
- AI artifacts created in: `docs/` directory

## Conclusion

The vdf-ml repository has clear AI accessibility challenges primarily around domain ambiguity and terminology confusion. Created 5 minimal AI artifacts addressing these issues. Identified cleanup candidates, starting with deprecated-code verification -- an earlier version of this conclusion listed "data format bridge" first, which was a misdiagnosis (see correction above); that item has been removed, not just reprioritized.

**Ready for**: Code cleanup phase starting with highest priority candidates.

## Current Documentation Assessment

### Strengths
- **Comprehensive core documentation:** README.md, PIPELINE.md, SCHEMA.md, TESTING.md, PCA_GUIDE.md provide excellent conceptual coverage
- **Function indices:** PHYSICS.md, DATA_PROC.md, LABELING.md, ML_MODELS.md, DEPRECATED.md effectively catalog available functionality
- **Clear terminology:** README's "cluster_phys vs cluster_ml" distinction prevents common confusion
- **Testing focus:** TESTING.md provides practical verification workflows with concrete examples
- **Cognitive design:** Documentation follows progressive disclosure and emphasizes physical concepts over code

### Critical Gaps

#### 1. **Missing Getting Started Guide**
- No installation/requirements documentation
- No setup instructions beyond basic venv activation
- No dependency management documentation (requirements.txt, pip, conda)
- No guidance on obtaining/using the analysator library

#### 2. **Configuration Documentation Deficit**
- 18 YAML config files with minimal inline documentation
- Parameters lack explanations of effects, units, or valid ranges
- No configuration reference guide
- `pipeline_config.py` contains critical constants with sparse comments

#### 3. **Workflow Examples Missing**
- No complete end-to-end examples showing actual command sequences
- Missing tutorial showing: setup → extract → label → PCA → plot workflow
- No example notebooks for common analysis tasks
- Existing notebooks are exploratory, not instructional

#### 4. **API Documentation Incomplete**
- Function indices lack parameter documentation
- Docstring coverage inconsistent (PIPELINE.md discourages comprehensive docstrings)
- No examples of common usage patterns
- No cross-references between related functions

#### 5. **Architecture Decisions Undocumented**
- Why certain design choices were made (e.g., two PCA pipelines)
- History of removed features (boundary_layer, margin_di)
- Tradeoffs between different feature representations
- Performance considerations and optimization strategies

#### 6. **Developer Onboarding Lacking**
- No contributor guide
- Missing code review guidelines
- No documentation on extending the system (adding new substances)
- Testing strategy documented but not development workflow

## User Needs Analysis

Based on codebase complexity, users need:

### **New Researchers/Students**
- **Need:** Quick start, minimal friction to run basic analysis
- **Missing:** Installation guide, simple example, dependency management
- **Impact:** High friction to entry, likely to abandon or misuse

### **Experienced Physicists**
- **Need:** Understand physical parameters, configure detectors, interpret results
- **Missing:** Configuration reference, parameter effects documentation
- **Impact:** Misconfigured analyses, wasted computation time

### **ML Practitioners**
- **Need:** Feature engineering details, model training workflows, PCA interpretation
- **Missing:** End-to-end training examples, feature representation comparisons
- **Impact:** Suboptimal model performance, misunderstanding of results

### **Maintainers/Contributors**
- **Need:** Architecture understanding, extension patterns, testing workflows
- **Missing:** Architecture decision records, contributor guide
- **Impact:** Codebase entropy, difficulty extending functionality

## Documentation Improvement Approaches

### Level 1: Minimal (Lowest Effort)
1. **Add installation guide** (requirements.txt, setup.py, or pyproject.toml)
2. **Document configuration parameters** with inline comments in configs/
3. **Create one end-to-end example** in examples/ or as a notebook
4. **Add docstrings to key public API functions**

**Effort:** 2-3 days

### Level 2: Comprehensive (Recommended)
1. **Complete Level 1** plus:
2. **Configuration reference guide** documenting all parameters
3. **Workflow tutorials** for common analysis patterns
4. **API documentation** with Sphinx/autodoc or improved function indices
5. **Architecture decision records** for key design choices
6. **Contributor guide** with development workflow

**Effort:** 1-2 weeks

### Level 3: Professional (Full Documentation Suite)
1. **Complete Level 2** plus:
2. **Interactive documentation** (Jupyter Book, ReadTheDocs)
3. **API reference** with searchable documentation
4. **Video tutorials** for complex workflows
5. **Cheat sheets** for common tasks
6. **Integration with testing** (doctests, example-based testing)
7. **Documentation CI/CD** (build and deploy on changes)

**Effort:** 3-4 weeks

## Scope Estimation

### Immediate Priority Tasks
1. **Getting Started Guide** (1 day): `docs/getting-started.md`
2. **Configuration Reference** (2 days): `docs/config-reference.md`
3. **Workflow Examples** (2 days): `examples/basic-workflow.ipynb`
4. **API Docstrings** (1 day): Add to 20-30 key functions

### Medium-term Improvements
1. **Architecture Decision Records** (2 days): `docs/decisions/`
2. **Contributor Guide** (1 day): `docs/contributing.md`
3. **Function Parameter Documentation** (3 days): Enhance existing indices

### Long-term Vision
1. **Interactive Documentation** (1 week): Jupyter Book
2. **Documentation CI/CD** (2 days): GitHub Actions workflow
3. **Video Tutorials** (1 week): Screen recordings with narration

## Risks

### Technical Risks
- **Documentation drift:** Documentation not updated with code changes
- **Over-documentation:** Creating documentation that's never read
- **Maintenance burden:** Additional documentation to maintain
- **Inconsistent style:** Different contributors using different styles

### Mitigation Strategies
1. **Integration with development workflow:** Document as part of PR process
2. **Lightweight approach:** Focus on practical, frequently-needed documentation
3. **Automated validation:** CI checks for broken links, missing docstrings
4. **Style guide:** Documentation conventions in `PIPELINE.md`

### Skill Resolution
- **Physics domain knowledge:** Already covered by existing skills
- **ML domain knowledge:** Already covered by existing skills
- **Documentation skills:** Covered by cognitive-doc-design skill
- **Repository structure knowledge:** Covered by orchestrate skill

## Recommendations

### Next Phase: Propose
Create a structured proposal for documentation improvements focusing on:

1. **Priority areas:** Getting started, configuration reference, workflow examples
2. **Implementation approach:** Minimal viable documentation first
3. **Success metrics:** Reduced onboarding time, fewer support questions
4. **Integration strategy:** Documentation as part of development workflow

### Key Design Principles to Maintain
- **Cognitive load reduction:** Continue progressive disclosure pattern
- **Physicists first:** Emphasize physical concepts over code
- **Multi-user friendly:** Clear for concurrent contributors
- **Practical over comprehensive:** Focus on what users actually need

### Artifact Storage
- Exploration saved to: `openspec/changes/improve-documentation/explore.md`
- Topic key for memory: `sdd/improve-documentation/explore`

## Conclusion

The vdf-ml repository has excellent core documentation but critical gaps in practical usage documentation. A focused improvement targeting **getting started guides**, **configuration documentation**, and **workflow examples** would dramatically improve usability without excessive maintenance burden. The repository's existing documentation patterns (cognitive design, physicist-first approach) provide a strong foundation to build upon.

**Next recommended phase:** propose