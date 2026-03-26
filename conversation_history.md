# Conversation History - March 24, 2026

## Session Summary
Reviewed RL training runs across multiple folders and restarted broke training sessions.

---

## Key Updates & Findings

### RL_FOLDER_VS2 Status
- **Stage 7 Runs** (Conservative, Moderate, Aggressive): All three halted at ~iteration 800-802 due to computer restart overnight
- **Status**: Resumed all three runs from checkpoint_800.pt
  - Conservative: 74% SR (6% short of 80% threshold)
  - Moderate: 62-64% SR
  - Aggressive: 63% SR

### VS3_checkpoints (Root Directory)
- **Stage 1 Complete** for all three learning rates
  - Conservative: 214 iterations, 704 episodes → 80% SR
  - Moderate: 123 iterations, 246 episodes → 80% SR ✅ **Most efficient**
  - Aggressive: 155 iterations, 326 episodes → 80% SR

### RL_FOLDER_VS3 Discovery
- Found **two separate VS3 checkpoint directories**:
  - Root: `/VS3_checkpoints/` - Stage 1 only
  - Nested: `/Experiments/Cole/RL_FOLDER_VS3/VS3_checkpoints/` - Different training

### Stage 2 Training Issues Found
- Aggressive, Conservative, Moderate logs in RL_FOLDER_VS3 showed **suspicious Stage 2→3 jump** in only 2 iterations
- Logs were incomplete (~80 lines) instead of full training history
- Detected curriculum bug: jumped from "STARTING TRAINING FROM STAGE 2" directly to "Stage 3/7: Object Pushing Training"
- **Decision**: Started proper Stage 2 training from stage_1_complete.pt checkpoints with 150 iterations

### Current Training Status (March 24, 2026)

#### Stage 7 Resumption (RL_FOLDER_VS2)
| Run | Terminal ID | Status | Last SR |
|-----|------------|--------|---------|
| Conservative | 59a6f4bc-... | ▶️ Running | 74% |
| Moderate | 8b969b91-... | ▶️ Running | 62-64% |
| Aggressive | f3da3737-... | ▶️ Running | 63% |

#### Stage 2 Training (RL_FOLDER_VS3) - Just Started
| Run | Terminal ID | Config | Iterations | Status |
|-----|------------|--------|-----------|---------|
| Aggressive | 072e9937-... | nav_config_aggressive.yaml | 150 | ▶️ Starting |
| Conservative | 40739913-... | nav_config_conservative.yaml | 150 | ▶️ Starting |
| Moderate | 300f167c-... | nav_config_moderate.yaml | 150 | ▶️ Starting |

---

## Tasks Completed
✅ Reviewed RL_FOLDER_VS2 runs (Stage 5 & Stage 7)  
✅ Analyzed VS3 checkpoint runs (Stage 1)  
✅ Discovered Stage 2 training bug in RL_FOLDER_VS3  
✅ Resumed Stage 7 runs from checkpoint_800.pt  
✅ Started Stage 2: Enhanced Stability training (150 iterations each)  

---

## Next Steps
- Monitor Stage 2 training convergence (target: 80% SR)
- Continue Stage 7 resumption until 80% threshold reached
- Once Stage 2 complete, advance to Stage 3
- Compare learning rates across stages

---

## Notes
- **Optimal Learning Rate (Stage 1)**: Moderate (1.0e-4) - fastest convergence
- **Stage 7 Bottleneck**: Heavy obstacles significantly harder than earlier stages
- **Stage 2 Config**: 5-stage curriculum (not 8-stage like RL_FOLDER_VS2)
  - Stage 1: Stability Foundation
  - Stage 2: Enhanced Stability (10% all obstacle types)
  - Stage 3: Object Pushing Training
  - Stage 4-5: Navigation and higher difficulty tasks
