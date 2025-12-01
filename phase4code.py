#!/usr/bin/env python3
"""
phase4code_fixed.py

Gemini-enabled WhatIf + Intervention agents (Phase 4 & 5)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, List, Any, Literal, Tuple
import pandas as pd
import numpy as np
import textwrap
import os
import json

# -------------------------
# Robust Gemini LLM client
# -------------------------
try:
    from google.genai import Client as GenaiClient
    from google.genai.types import GenerateContentConfig
    GENAI_NEW_SDK = True
except Exception:
    GenaiClient = None
    GenerateContentConfig = None
    GENAI_NEW_SDK = False

try:
    from google import genai as genai_pkg
    GENAI_OLD_PKG = True
except Exception:
    genai_pkg = None
    GENAI_OLD_PKG = False

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "models/gemini-2.5-flash", max_output_tokens: int = 3000, temperature: float = 0.2):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.client = None
        self._client_type = None

        if not self.api_key:
            print("[LLMClient] Warning: no GEMINI API key provided; LLM disabled.")
            return

        # Try new SDK
        try:
            if GENAI_NEW_SDK and GenaiClient is not None:
                self.client = GenaiClient(api_key=self.api_key)
                self._client_type = "genai_new"
                return
        except Exception as e:
            print("[LLMClient] new genai.Client init failed:", e)

        # Try older package
        try:
            if GENAI_OLD_PKG and genai_pkg is not None:
                try:
                    genai_pkg.configure(api_key=self.api_key)
                except Exception:
                    pass
                self.client = genai_pkg
                self._client_type = "genai_old"
                return
        except Exception as e:
            print("[LLMClient] old genai init failed:", e)

        print("[LLMClient] No working genai client available; LLM disabled.")

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate(self, prompt: Optional[str]) -> str:
        """
        Generate text for prompt. Always returns a Python string (never None).
        If the LLM client is not configured, returns a helpful preview string.
        """
        if prompt is None:
            return "[LLMClient] Error: generate(prompt) called with prompt=None"

        if not getattr(self, "client", None):
            return (
                "LLM provider not configured or unavailable. Prompt I would send:\n\n"
                + textwrap.shorten(prompt, width=1000, placeholder=" ...")
            )

        # helper to extract textual output robustly
        def extract_text(resp) -> str:
            if resp is None:
                return ""
            # direct string
            if isinstance(resp, str):
                return resp
            # .text attribute (common)
            if hasattr(resp, "text"):
                try:
                    return str(resp.text)
                except Exception:
                    pass
            # try to_dict() then search for common keys
            try:
                d = resp.to_dict() if hasattr(resp, "to_dict") else None
            except Exception:
                d = None
            if isinstance(d, dict):
                for k in ("text", "output", "content", "results", "candidates", "outputs"):
                    if k in d:
                        val = d[k]
                        # list handling
                        if isinstance(val, list) and len(val) > 0:
                            first = val[0]
                            if isinstance(first, dict):
                                for kk in ("text", "content"):
                                    if kk in first:
                                        return str(first[kk])
                            return str(first)
                        return str(val)
                # nested candidates handling
                if "candidates" in d and isinstance(d["candidates"], list) and len(d["candidates"]) > 0:
                    c = d["candidates"][0]
                    if isinstance(c, dict):
                        for kk in ("content", "text"):
                            if kk in c:
                                return str(c[kk])
                    return str(c)
                # try outputs list
                if "outputs" in d and isinstance(d["outputs"], list) and len(d["outputs"]) > 0:
                    out0 = d["outputs"][0]
                    if isinstance(out0, dict):
                        for kk in ("content", "text"):
                            if kk in out0:
                                return str(out0[kk])
                    return str(out0)
            # as a last resort, try str(resp)
            try:
                return str(resp)
            except Exception:
                return ""

        # We'll attempt multiple SDK call shapes. For each, if we get resp, we extract text
        last_resp = None
        # 1) new SDK: client.models.generate_content(...)
        try:
            if self._client_type == "genai_new" and hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                contents = [prompt]
                try:
                    if GenerateContentConfig is not None:
                        cfg = GenerateContentConfig(max_output_tokens=self.max_output_tokens, temperature=self.temperature)
                        last_resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
                    else:
                        last_resp = self.client.models.generate_content(model=self.model, contents=contents, max_output_tokens=self.max_output_tokens)
                except TypeError:
                    last_resp = self.client.models.generate_content(model=self.model, contents=prompt)
                out = extract_text(last_resp)
                if out:
                    return out
        except Exception as e:
            # keep going to other fallbacks
            last_exc = e

        # 2) older genai package entry points
        try:
            if self._client_type == "genai_old":
                if hasattr(self.client, "generate_content"):
                    try:
                        last_resp = self.client.generate_content(model=self.model, prompt=prompt)
                    except TypeError:
                        last_resp = self.client.generate_content(prompt)
                    out = extract_text(last_resp)
                    if out:
                        return out
                if hasattr(self.client, "generate"):
                    try:
                        last_resp = self.client.generate(model=self.model, prompt=prompt)
                    except TypeError:
                        last_resp = self.client.generate(self.model, prompt)
                    out = extract_text(last_resp)
                    if out:
                        return out
        except Exception:
            pass

        # 3) generic patterns
        try:
            if hasattr(self.client, "generate_text"):
                try:
                    last_resp = self.client.generate_text(model=self.model, prompt=prompt, max_output_tokens=self.max_output_tokens)
                except TypeError:
                    last_resp = self.client.generate_text(self.model, prompt)
                out = extract_text(last_resp)
                if out:
                    return out
            if hasattr(self.client, "generate"):
                try:
                    last_resp = self.client.generate(model=self.model, prompt=prompt, max_output_tokens=self.max_output_tokens)
                except TypeError:
                    last_resp = self.client.generate(self.model, prompt)
                out = extract_text(last_resp)
                if out:
                    return out
        except Exception:
            pass

        # If we got here, extraction returned empty string for any resp we received.
        # Log repr of the last_resp for diagnosis and return a stable fallback string.
        try:
            debug_repr = repr(last_resp)[:2000]
            print("[LLMClient DEBUG] Last SDK response repr (truncated):", debug_repr)
        except Exception:
            print("[LLMClient DEBUG] Could not repr last_resp")

        return "[LLMClient] All Gemini call attempts returned no extractable text. Prompt I would send:\n\n" + textwrap.shorten(prompt, width=1200, placeholder=" ...")


# -------------------------
# Data classes
# -------------------------
@dataclass
class ScenarioChange:
    targets: Dict[str, float]
    deltas: Dict[str, float]


@dataclass
class ScenarioConfig:
    level: Literal["campus", "campus_group"]
    id_value: Any
    year: Optional[int] = None


@dataclass
class ScenarioResult:
    baseline_row: pd.Series
    scenario_row: pd.Series
    changed_columns: List[str]
    description: str

# -------------------------
# WhatIfAgent
# -------------------------
class WhatIfAgent:
    def __init__(self, df: pd.DataFrame, id_cols: Dict[str, str], risk_index_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None, prediction_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None, llm_client: Optional[LLMClient] = None):
        self.df = df.copy()
        self.id_cols = id_cols
        self.risk_index_fn = risk_index_fn or self._default_risk_index
        self.prediction_fn = prediction_fn
        self.llm = llm_client

    def run_scenario(self, config: ScenarioConfig, change: ScenarioChange, scenario_description: str = "") -> Tuple[ScenarioResult, Optional[str]]:
        base_row_idx = self._locate_row(config)
        if base_row_idx is None:
            raise ValueError("Could not locate row for scenario config.")

        baseline_row = self.df.loc[base_row_idx].copy()
        df_scenario = self.df.copy()
        scenario_row = df_scenario.loc[base_row_idx].copy()

        changed_cols = []

        for col, target_val in change.targets.items():
            if col in df_scenario.columns:
                df_scenario.at[base_row_idx, col] = target_val
                scenario_row[col] = target_val
                changed_cols.append(col)
            else:
                print(f"[WhatIfAgent] Warning: target column '{col}' not found in dataframe.")

        for col, delta_val in change.deltas.items():
            if col in df_scenario.columns:
                existing = df_scenario.at[base_row_idx, col]
                existing_num = float(existing) if pd.notna(existing) and str(existing) != "nan" else 0.0
                new_val = existing_num + float(delta_val)
                df_scenario.at[base_row_idx, col] = new_val
                scenario_row[col] = new_val
                if col not in changed_cols:
                    changed_cols.append(col)
            else:
                print(f"[WhatIfAgent] Warning: delta column '{col}' not found in dataframe.")

        if all(c in df_scenario.columns for c in ("ERW_SAT", "Math_SAT")) and "Total_SAT" in df_scenario.columns:
            try:
                df_scenario.at[base_row_idx, "Total_SAT"] = float(df_scenario.at[base_row_idx, "ERW_SAT"] or 0) + float(df_scenario.at[base_row_idx, "Math_SAT"] or 0)
            except Exception:
                pass

        try:
            df_scenario["Risk_Index_0_100"] = self.risk_index_fn(df_scenario)
        except Exception as e:
            print(f"[WhatIfAgent] Risk index recompute failed: {e}")
            df_scenario["Risk_Index_0_100"] = df_scenario.get("risk_score", np.nan)

        if self.prediction_fn is not None:
            try:
                df_scenario = self.prediction_fn(df_scenario)
            except Exception as e:
                print(f"[WhatIfAgent] prediction_fn failed: {e}")

        scenario_row = df_scenario.loc[base_row_idx].copy()

        result = ScenarioResult(baseline_row=baseline_row, scenario_row=scenario_row, changed_columns=changed_cols, description=scenario_description)
        narrative = None
        if self.llm is not None:
            try:
                narrative = self._summarize_with_llm(config, result)
                # treat explicit "None" or empty responses as missing
                if narrative is None:
                    narrative = None
                else:
                    # normalize string outputs
                    narrative_str = str(narrative).strip()
                    if narrative_str == "" or narrative_str.lower() == "none":
                        narrative = None
                    else:
                        narrative = narrative_str
            except Exception as e:
                narrative = f"[WhatIfAgent] LLM summarization failed: {e}"

        # FIX: The missing return statement has been added here.
        return result, narrative


    def _locate_row(self, config: ScenarioConfig) -> Optional[Any]:
        df = self.df
        year_col = self.id_cols.get("year")
        campus_col = self.id_cols.get("campus")
        group_col = self.id_cols.get("group")

        required = [c for c in (campus_col, group_col, year_col) if c]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"[WhatIfAgent._locate_row] Missing expected columns: {missing}. Check id_cols mapping.")

        def try_masks(mask):
            cand = df[mask]
            return cand if not cand.empty else None

        candidates = pd.DataFrame()
        if config.level == "campus":
            try:
                mask = df[campus_col] == config.id_value
                res = try_masks(mask)
                if res is not None:
                    candidates = res
                else:
                    mask2 = df[campus_col].astype(str).str.strip() == str(config.id_value).strip()
                    res2 = try_masks(mask2)
                    if res2 is not None:
                        candidates = res2
                    else:
                        try:
                            val_num = pd.to_numeric(config.id_value, errors="coerce")
                            if pd.notna(val_num):
                                mask3 = pd.to_numeric(df[campus_col], errors="coerce") == val_num
                                res3 = try_masks(mask3)
                                if res3 is not None:
                                    candidates = res3
                        except Exception:
                            pass
            except Exception:
                mask2 = df[campus_col].astype(str).str.strip() == str(config.id_value).strip()
                candidates = df[mask2]
        else:
            campus_val, group_val = config.id_value
            try:
                mask = (df[campus_col] == campus_val) & (df[group_col] == group_val)
                res = try_masks(mask)
                if res is not None:
                    candidates = res
                else:
                    mask2 = (df[campus_col].astype(str).str.strip() == str(campus_val).strip()) & (df[group_col].astype(str).str.strip() == str(group_val).strip())
                    res2 = try_masks(mask2)
                    if res2 is not None:
                        candidates = res2
                    else:
                        try:
                            val_num = pd.to_numeric(campus_val, errors="coerce")
                            if pd.notna(val_num):
                                mask3 = (pd.to_numeric(df[campus_col], errors="coerce") == val_num) & (df[group_col].astype(str).str.strip() == str(group_val).strip())
                                res3 = try_masks(mask3)
                                if res3 is not None:
                                    candidates = res3
                        except Exception:
                            pass
            except Exception:
                mask2 = (df[campus_col].astype(str).str.strip() == str(campus_val).strip()) & (df[group_col].astype(str).str.strip() == str(group_val).strip())
                candidates = df[mask2]

        if config.year is not None and year_col in df.columns:
            candidates = candidates[candidates[year_col] == config.year]

        if candidates.empty:
            sample_vals = df[campus_col].dropna().unique()[:20].tolist()
            sample_preview = [str(x) for x in sample_vals[:10]]
            raise ValueError(f"[WhatIfAgent._locate_row] No matching rows for {config}. Sample campus ids: {sample_preview}")

        if config.year is None and year_col in df.columns:
            latest_year = candidates[year_col].max()
            candidates = candidates[candidates[year_col] == latest_year]

        return candidates.index[0]

    def _default_risk_index(self, df: pd.DataFrame) -> pd.Series:
        cols = {"sat_erw": "ERW_SAT", "sat_math": "Math_SAT", "tsi": "Above_TSI_Both_Rate", "part": "Part_Rate", "trend": "Total_SAT_lag1", "gap": "Max_Equity_Gap"}
        def col_or_zero(c):
            return pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else pd.Series(0, index=df.index)
        sat_erw = col_or_zero(cols["sat_erw"])
        sat_math = col_or_zero(cols["sat_math"])
        tsi = col_or_zero(cols["tsi"])
        part = col_or_zero(cols["part"])
        trend = col_or_zero(cols["trend"])
        gap = col_or_zero(cols["gap"])
        def norm(s, invert=False):
            s = s.replace([np.inf, -np.inf], np.nan).fillna(s.median() if s.median() == s.median() else 0)
            if s.max() == s.min():
                return pd.Series(0.5, index=s.index)
            r = (s - s.min()) / (s.max() - s.min())
            return 1 - r if invert else r
        sat_erw_r = norm(sat_erw, invert=True)
        sat_math_r = norm(sat_math, invert=True)
        tsi_r = norm(tsi, invert=True)
        part_r = norm(part, invert=True)
        trend_r = norm(-trend)
        gap_r = norm(gap)
        risk = 0.2 * sat_erw_r + 0.2 * sat_math_r + 0.2 * tsi_r + 0.15 * part_r + 0.15 * trend_r + 0.1 * gap_r
        return (risk * 100).clip(0, 100)

    def _summarize_with_llm(self, config: ScenarioConfig, result: ScenarioResult) -> str:
        if not self.llm:
            return "LLM disabled (no API key configured)."
        campus_col = self.id_cols.get("campus", "AICode")
        group_col = self.id_cols.get("group", "Group")
        year_col = self.id_cols.get("year", "year")
        b = result.baseline_row
        s = result.scenario_row
        campus_id = b.get(campus_col, "N/A")
        group = b.get(group_col, "All Students")
        year = int(b.get(year_col)) if year_col in b.index else "latest"
        change_lines = []
        for col in result.changed_columns:
            if col in b.index and col in s.index:
                try:
                    before = float(b[col]); after = float(s[col])
                    change_lines.append(f"{col}: {before:.2f} → {after:.2f} (Δ {after-before:+.2f})")
                except Exception:
                    change_lines.append(f"{col}: {b.get(col)} → {s.get(col)}")
        if "risk_score" in b.index and "risk_score" in s.index:
            try:
                br = float(b["risk_score"]); ar = float(s["risk_score"])
                change_lines.append(f"risk_score: {br:.2f} → {ar:.2f} (Δ {ar-br:+.2f})")
            except:
                pass
        metric_block = "\n".join(change_lines) if change_lines else "No metric changes detected."
        prompt = f"""
You are an academic data analyst. Summarize this what-if scenario for a superintendent.

Campus ID: {campus_id}
Group: {group}
Year: {year}

Scenario description:
{result.description}

Metric changes:
{metric_block}

Write:
- 2 short paragraphs
- Explain what changed and why it matters
- Mention impact on student outcomes and equity
- Clear, superintendent-level language
""".strip()
        return self.llm.generate(prompt)

# -------------------------
# InterventionAgent
# -------------------------
class InterventionAgent:
    def __init__(self, llm_client: Optional[LLMClient], id_cols: Dict[str, str]):
        self.llm = llm_client
        self.id_cols = id_cols

    def recommend_for_row(self, row: pd.Series, scenario_result: Optional[ScenarioResult] = None) -> str:
        campus_col = self.id_cols.get("campus", "AICode")
        group_col = self.id_cols.get("group", "Group")
        year_col = self.id_cols.get("year", "year")

        campus_id = row.get(campus_col, "N/A")
        group = row.get(group_col, "All Students")
        year = int(row.get(year_col)) if year_col in row.index else "latest"

        risk = float(row.get("Risk_Index_0_100", row.get("risk_score", np.nan)) or np.nan)
        sat_erw = float(row.get("ERW_SAT", np.nan) or np.nan) if "ERW_SAT" in row.index else np.nan
        sat_math = float(row.get("Math_SAT", np.nan) or np.nan) if "Math_SAT" in row.index else np.nan
        tsi = float(row.get("Above_TSI_Both_Rate", np.nan) or np.nan) if "Above_TSI_Both_Rate" in row.index else np.nan
        part = float(row.get("Part_Rate", np.nan) or np.nan) if "Part_Rate" in row.index else np.nan
        max_gap = float(row.get("Max_Equity_Gap", np.nan) or np.nan) if "Max_Equity_Gap" in row.index else np.nan

        scenario_text = ""
        if scenario_result is not None:
            scenario_text = _build_scenario_diff_text(scenario_result)


        prompt = f"""
You are an AI assistant helping a school district design academic interventions. Your output MUST be concise.

Context:
- Campus ID: {campus_id}
- Student Group: {group}
- Year: {year}

Key metrics (for diagnosis):
- Risk Index (0-100): {risk:.2f}
- SAT Math avg: {sat_math:.2f}
- TSI readiness (%): {tsi:.2f}
- Max equity gap (pp): {max_gap:.2f}

Task:
1) Identify the 2 highest **priority problems** based on the metrics.
2) For each problem, propose **one specific intervention**.
3) Describe the **intervention** and its **expected effect** in **one single sentence** each.

Return the result as a simple, single-level JSON object (no markdown):

{{
  "recommendations": [
    {{
      "problem": "Low TSI readiness (TSI < 50%)",
      "intervention": "Launch a 6-week intensive TSI reading/writing bootcamp with daily practice.",
      "effect": "Increase TSI readiness rate by 10 percentage points within one semester."
    }},
    {{
      "problem": "Large Equity Gap (Max Gap > 10pp)",
      "intervention": "Implement targeted small-group tutoring for the lowest performing subgroup.",
      "effect": "Reduce the largest achievement gap by 5 percentage points by year end."
    }}
  ],
  "summary_for_leaders": "Two high-priority interventions targeting college readiness and equity gaps."
}}

""".strip()

        # Robust LLM usage: try LLM, log its return; fall back if missing/empty/exception
        if self.llm:
            try:
                llm_out = self.llm.generate(prompt)
                # debug info
                print("[InterventionAgent DEBUG] llm.generate() returned type:", type(llm_out), "len:", (len(llm_out) if isinstance(llm_out, (str, list, dict)) else "n/a"))
                # normalize to string if possible
                if isinstance(llm_out, str):
                    llm_text = llm_out.strip()
                else:
                    llm_text = str(llm_out).strip() if llm_out is not None else ""
                # treat empty/"None" as no response
                if llm_text and llm_text.lower() != "none":
                    return llm_text
                else:
                    print("[InterventionAgent DEBUG] LLM returned empty/'None' output; falling back to rule-based recommendations.")
            except Exception as e:
                print("[InterventionAgent DEBUG] LLM call raised exception:", repr(e), "; falling back to deterministic rules.")

        # deterministic fallback
        problems = []
        interventions = []
        if pd.isna(sat_math) or sat_math < 450:
            problems.append({"name": "Low SAT Math", "why_important": "Math affects readiness", "metrics": ["Math_SAT"]})
            interventions.append({
                "title": "Math tutoring bootcamp",
                "target_problem": "Low SAT Math",
                "description": "Targeted Algebra II tutoring, 3x/week for 10 weeks.",
                "expected_effect": "Increase Math_SAT by 10-25 points",
                "time_horizon": "short-term"
            })
        if pd.isna(tsi) or tsi < 50:
            problems.append({"name": "Low TSI readiness", "why_important": "TSI predicts college remediation", "metrics": ["Above_TSI_Both_Rate"]})
            interventions.append({
                "title": "TSI bootcamp",
                "target_problem": "Low TSI readiness",
                "description": "6-week TSI reading/writing bootcamp with practice tests.",
                "expected_effect": "Increase TSI readiness by ~10 percentage points",
                "time_horizon": "short-term"
            })
        if max_gap and max_gap > 10:
            problems.append({"name": "Large equity gap", "why_important": "Indicates unequal outcomes", "metrics": ["Max_Equity_Gap"]})
            interventions.append({
                "title": "Equity-focused small groups",
                "target_problem": "Large equity gap",
                "description": "Targeted culturally responsive tutoring for underperforming subgroups.",
                "expected_effect": "Reduce gap by several percentage points",
                "time_horizon": "medium-term"
            })

        summary = "Rule-based fallback interventions. Use LLM for richer, context-aware plans."
        return str({"priority_problems": problems, "interventions": interventions, "summary_for_leaders": summary})

def _build_scenario_diff_text(result: ScenarioResult) -> str:
    b = result.baseline_row
    s = result.scenario_row
    lines = []
    for col, pct in [("ERW_SAT", False), ("Math_SAT", False), ("Total_SAT", False),
                     ("Above_TSI_Both_Rate", True), ("Part_Rate", True), ("risk_score", False), ("Max_Equity_Gap", True)]:
        if col in b.index and col in s.index:
            before = b[col]
            after = s[col]
            if not (pd.isna(before) and pd.isna(after)):
                unit = "%" if pct else ""
                try:
                    lines.append(f"{col}: {float(before):.2f}{unit} -> {float(after):.2f}{unit} (Δ {float(after)-float(before):+.2f}{unit})")
                except Exception:
                    lines.append(f"{col}: {before} -> {after}")
    if not lines:
        return "No scenario differences computed."
    return "Scenario changes (before -> after):\n" + "\n".join(lines)

# -------------------------
# Main demo runner
# -------------------------
if __name__ == "__main__":
    DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    
    # === START OF API KEY CONFIGURATION ===
    # PLACEHOLDER: Replace "YOUR_ACTUAL_GEMINI_API_KEY_HERE" with your key.
    HARDCODED_API_KEY = "AIzaSyDX53CvdCNwZL60Aa7Bhi7oMKqaieGyz7k" 
    
    api_key_for_client = HARDCODED_API_KEY
    
    # Fallback to environment variable if the hardcoded one is still the placeholder.
    if api_key_for_client == "AIzaSyDX53CvdCNwZL60Aa7Bhi7oMKqaieGyz7k":
        api_key_for_client = os.getenv("GEMINI_API_KEY")

    # instantiate LLM client instance (pass the selected key)
    try:
        # Pass the key directly to LLMClient
        llm_client = LLMClient(api_key=api_key_for_client, model=DEFAULT_MODEL)
    except Exception as e:
        print("[main] LLMClient init failed:", e)
        llm_client = None
    # === END OF API KEY CONFIGURATION ===


    # candidate files (modify paths if needed)
    candidate_paths = [
        "/Users/nehal/Desktop/phase3_predictions.xlsx",
        "/Users/nehal/Desktop/phase2_analytics_with_ACT.xlsx",
        "/Users/nehal/Desktop/combined_year_dataset_cleaned.xlsx",
        "/mnt/data/phase3_predictions.xlsx",
        "/mnt/data/phase2_analytics_with_ACT.xlsx",
        "/mnt/data/combined_year_dataset_cleaned.xlsx"
    ]

    df_master = pd.DataFrame()
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                df_master = pd.read_excel(p)
                print(f"[main] Loaded '{p}' shape={df_master.shape}")
                break
            except Exception as e:
                print(f"[main] Could not load '{p}': {e}")

    if df_master.empty:
        raise SystemExit("[main] ERROR: No Phase2/3 files loaded. Place them in one of the candidate paths and re-run.")

    # print some columns
    print("\n[main] Columns (first 200):")
    for i, c in enumerate(df_master.columns[:200]):
        print(f"  {i:03d}: {c!r}")

    # detect id columns
    def find_first(cols, pats):
        for pat in pats:
            for c in cols:
                if pat in c.lower():
                    return c
        return None

    cols = list(df_master.columns)
    campus_col = find_first(cols, ["aicode", "aic", "campus_id", "campus", "camp", "camp_code", "campid", "school"])
    group_col = find_first(cols, ["group", "subgroup", "student_group", "sub_group"])
    year_col = find_first(cols, ["year", "acadyear", "academic_year", "fy"])

    id_cols = {"campus": campus_col or "AICode", "group": group_col or "Group", "year": year_col or "year"}
    print("\n[main] Auto-detected id_cols (change if wrong):")
    print(id_cols)

    if id_cols["campus"] in df_master.columns:
        campus_col_name = id_cols["campus"]
        sample_ids = df_master[campus_col_name].dropna().unique()[:20].tolist()
        print(f"\n[main] Sample campus ids (first 20): {sample_ids}")
    else:
        print(f"[main] WARNING: campuses column '{id_cols['campus']}' not in DF columns. Update id_cols manually.")

    # instantiate agents (pass instance, not class)
    what_if_agent = WhatIfAgent(df=df_master, id_cols=id_cols, risk_index_fn=None, prediction_fn=None, llm_client=llm_client)
    intervention_agent = InterventionAgent(llm_client=llm_client, id_cols=id_cols)

    # choose example campus
    if id_cols["campus"] in df_master.columns:
        campus_col_name = id_cols["campus"]
        raw_val = df_master[campus_col_name].dropna().unique()[0]
        col_dtype = df_master[campus_col_name].dtype
        example_campus = raw_val
        try:
            if pd.api.types.is_integer_dtype(col_dtype):
                example_campus = int(raw_val)
            elif pd.api.types.is_float_dtype(col_dtype):
                example_campus = float(raw_val)
            else:
                example_campus = str(raw_val)
        except Exception:
            example_campus = raw_val

        print(f"[main] Using example campus id (column dtype={col_dtype}, value type={type(example_campus)}): {example_campus!r}")

        scenario_cfg = ScenarioConfig(level="campus", id_value=example_campus, year=None)
        scenario_change = ScenarioChange(targets={"Part_Rate": 90.0}, deltas={"Math_SAT": 20.0})

        try:
            scenario_result, narrative = what_if_agent.run_scenario(config=scenario_cfg, change=scenario_change, scenario_description="Demo: +20 Math_SAT & Part_Rate -> 90%")
            print("\n=== WHAT-IF: changed columns ===")
            print(scenario_result.changed_columns)

            print("\n=== WHAT-IF: baseline vs scenario snippet ===")
            for c in (scenario_result.changed_columns + ["Risk_Index_0_100", "Total_SAT", "Above_TSI_Both_Rate"]):
                if c in scenario_result.baseline_row.index:
                    before = scenario_result.baseline_row.get(c)
                    after = scenario_result.scenario_row.get(c)
                    print(f"{c}: {before} -> {after}")

            print("\n=== WHAT-IF: Narrative ===")
            print(narrative or "LLM narrative unavailable (no valid API key or client).")

            print("\n=== INTERVENTION RECOMMENDATIONS (Gemini LLM OR rule-based fallback) ===")
            rec = intervention_agent.recommend_for_row(scenario_result.scenario_row, scenario_result)
            print(rec)
        except Exception as e:
            print(f"[main] Could not run scenario: {e}")
    else:
        print("[main] Cannot run demo: campus id column missing. Edit id_cols and re-run.")