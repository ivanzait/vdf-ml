# AI Guides for vdf-ml

This folder contains AI-specific documentation to help LLM agents work effectively with the vdf-ml repository.

## Files

1. **`AI-AGENT-GUIDE.md`** - LLM-optimized repository overview
2. **`DECISION-ROUTING.md`** - Physics vs ML task classification guide  
3. **`KEY-CONCEPTS.md`** - Essential terms and relationships cheat sheet
4. **`AI-TASK-TEMPLATES.md`** - Common operation patterns with examples
5. **`AI-PITFALLS.md`** - Common mistakes and how to avoid them

## Purpose

These guides address AI-specific challenges in this cross-domain physics/ML repository:

- **Domain ambiguity** (physics vs ML tasks)
- **Terminology confusion** (`cluster_phys` vs `cluster_ml`)
- **Data format mismatches** (new vs old pipeline formats)
- **Architectural implicit knowledge** (layer separation, config systems)

## Design Principles

- **Token-efficient**: Optimized for LLM context windows
- **Action-oriented**: Clear next steps for different scenarios
- **Error-preventive**: Common pitfalls documented upfront
- **Skill-integrated**: Works with existing skill system (`orchestrate`)

## Usage for AI Agents

Start with `AI-AGENT-GUIDE.md` for overview, then:
- Use `DECISION-ROUTING.md` for task classification
- Reference `KEY-CONCEPTS.md` for terminology
- Follow `AI-TASK-TEMPLATES.md` for common workflows
- Check `AI-PITFALLS.md` to avoid common mistakes

## Generated

Created: 2026-07-22  
Purpose: Make vdf-ml AI-accessible for LLM agents