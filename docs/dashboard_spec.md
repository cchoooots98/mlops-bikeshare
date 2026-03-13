# Dashboard Specification

## Purpose
The dashboard must serve both business and operator users without hiding which prediction target is currently selected.

## Required Scope Selector
- target selector with:
  - `Bike stockout`
  - `Dock stockout`
- city selector may remain fixed to `paris` for this project
- environment indicator must be visible

## Required Status Cards
The top section must show:
- current target
- endpoint name
- model version
- decision threshold

## Required Sections
1. map of latest predictions
2. top-N risk table
3. station prediction history
4. model health
5. system health
6. data freshness

## Target-Aware Requirements
- no chart may hardcode `yhat_bikes` or `y_stockout_bikes_30`
- endpoint names must resolve from target plus environment
- CloudWatch queries must include `TargetName`
- prediction and quality prefixes must include `target=`
- switching target must update all cards, charts, and prefixes together

## Production Restrictions
- no debug-only publish controls in formal mode
- no target inference based only on endpoint name
- no legacy Athena `features_offline` dependency in formal dashboard paths

## Validation Checklist
- bikes view shows bikes label/score columns
- docks view shows docks label/score columns
- prod/staging endpoint names are correct
- model health charts use the selected target dimensions
- non-technical users can tell which target they are viewing within five seconds
