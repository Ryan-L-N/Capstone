"""
TRAINING TRACKING GUIDE
=======================
Comprehensive guide to monitoring Spot RL training progress.

Author: Autonomy Project
Date: February 2026
"""

# ============================================================================
# TRAINING TRACKING OVERVIEW
# ============================================================================

The training system provides FOUR complementary tracking methods:

1. TENSORBOARD (Real-Time Visualization)
   └─ Live training curves, loss plots, reward trends
   └─ Accessible via web browser during training
   └─ Best for monitoring training in progress

2. CSV LOGGING (Tabular Data Export)
   └─ All metrics saved to training_metrics.csv
   └─ Importable to Excel, pandas, or analysis tools
   └─ Best for data analysis and post-hoc comparison

3. CHECKPOINT SYSTEM (Model Snapshots)
   └─ Network weights saved every N episodes
   └─ Can resume interrupted training
   └─ Can evaluate policy at different stages
   └─ Best for model selection and evaluation

4. CONSOLE OUTPUT (Real-Time Feedback)
   └─ Rich formatted training metrics printed each logging interval
   └─ Visible in terminal during training
   └─ Best for quick feedback and debugging

# ============================================================================
# TENSORBOARD REAL-TIME MONITORING (RECOMMENDED FOR LIVE TRACKING)
# ============================================================================

SETUP (One-time):
  1. Install tensorboard if not already installed:
     pip install tensorboard

USAGE:
  1. Start training (opens window 1):
     cd c:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole
     C:\isaac-sim\python.bat SpotRL_Training.py --episodes 1000

  2. In a separate terminal (window 2), start TensorBoard:
     cd c:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole
     tensorboard --logdir ./runs/spot_rl --port 6006

  3. Open browser to:
     http://localhost:6006

WHAT YOU'LL SEE:
  ✓ Reward/avg_episode_reward    - Average episode reward over time
  ✓ Reward/max_episode_reward    - Best episode reward achieved
  ✓ Reward/min_episode_reward    - Worst episode reward
  ✓ Metrics/episode_length       - Average steps per episode
  ✓ Metrics/success_rate         - % of episodes reaching goal
  ✓ Loss/policy_loss             - Policy gradient loss
  ✓ Loss/value_loss              - Value function loss
  ✓ Loss/total_loss              - Combined training loss
  ✓ Training/total_steps         - Cumulative environment steps

KEY FEATURES:
  • Smooth curves (drag slider at bottom to adjust smoothing)
  • Download PNG images of plots
  • Cross-plot comparison tools
  • Real-time updates (refreshes automatically)
  • Multiple run comparison

TIPS:
  • Look for "UpTrend" in Reward: Policy learning to navigate better
  • Look for "DownTrend" in Loss: Training is converging
  • Entropy should gradually decrease: Exploration → exploitation
  • If reward plateaus: May need hyperparameter tuning
  • If loss diverges: May have learning rate instability


# ============================================================================
# CSV EXPORT & ANALYSIS (RECOMMENDED FOR POST-HOC ANALYSIS)
# ============================================================================

LOCATION:
  ./runs/spot_rl/training_metrics.csv

COLUMNS:
  episode            - Episode number (1-indexed)
  total_steps        - Cumulative environment steps
  avg_reward         - Average reward (last 10 episodes)
  avg_length         - Average episode length
  policy_loss        - PPO policy gradient loss
  value_loss         - Value function MSE loss
  total_loss         - Combined loss
  entropy            - Policy entropy (exploration)
  success_rate       - Fraction of successful episodes
  timestamp          - ISO format timestamp

USAGE IN EXCEL:
  1. Open Excel
  2. File → Open → Select training_metrics.csv
  3. Import as delimited (comma)
  4. Create charts from columns:
     • X-axis: episode
     • Y-axis: avg_reward, success_rate, policy_loss, etc.

USAGE IN PYTHON:
  import pandas as pd
  
  df = pd.read_csv('./runs/spot_rl/training_metrics.csv')
  
  # Plot reward trend
  df.plot(x='episode', y='avg_reward', figsize=(12, 6))
  
  # Statistics
  print(f"Final reward: {df['avg_reward'].iloc[-1]:.4f}")
  print(f"Success rate: {df['success_rate'].iloc[-1]:.1%}")
  print(f"Best reward: {df['avg_reward'].max():.4f}")

AUTOMATIC ANALYSIS:
  See section ANALYSIS TOOL below for automated plotting/reporting


# ============================================================================
# CHECKPOINT SYSTEM (FOR MODEL SELECTION & RESUMING)
# ============================================================================

LOCATION:
  ./checkpoints/spot_rl/

FILES CREATED:
  spot_rl_ep50.pt          - Network weights + optimizer (PyTorch)
  spot_rl_ep50.json        - Metadata (human-readable JSON)
  spot_rl_ep100.pt
  spot_rl_ep100.json
  ...
  spot_rl_final.pt         - Final trained model
  spot_rl_final.json

CHECKPOINT CONTENTS (PyTorch .pt file):
  {
    'episode': 50,
    'total_steps': 12500,
    'episode_rewards': [list of all episode rewards],
    'episode_lengths': [list of all episode lengths],
    'training_stats': {all loss/entropy history},
    'config': {training configuration used},
    'timestamp': '2026-02-16T10:30:45.123456',
    'policy_state_dict': {network parameters},
    'optimizer_state_dict': {Adam optimizer state}
  }

RESUMING INTERRUPTED TRAINING:
  Add to SpotRL_Training.py main():
  
  if args.resume:
      checkpoint = torch.load(args.resume)
      trainer.policy_net.load_state_dict(checkpoint['policy_state_dict'])
      trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      print(f"Resumed from episode {checkpoint['episode']}")

EVALUATING BEST MODEL:
  # Load best checkpoint based on reward
  df = pd.read_csv('./runs/spot_rl/training_metrics.csv')
  best_ep = df.loc[df['avg_reward'].idxmax(), 'episode']
  checkpoint = torch.load(f'./checkpoints/spot_rl/spot_rl_ep{int(best_ep)}.pt')
  
  # Deploy this model


# ============================================================================
# CONSOLE OUTPUT (REAL-TIME TERMINAL FEEDBACK)
# ============================================================================

EXAMPLE OUTPUT:
  ============================================================================================
  [Episode 10/1000] Training Progress
  ============================================================================================
    Reward:        145.2342 (max:    195.6234, min:     98.3421)
    Episode Length:    542.1 steps
    Success Rate:       40.0%
    Policy Loss:   0.234562
    Value Loss:    0.045123
    Total Loss:    0.128345
    Total Steps:        5420
  ============================================================================================

PRINTED EVERY:
  • 10 episodes (configurable via log_interval in config)
  • Scrolls in terminal, not stored separately

METRICS SHOWN:
  ✓ Recent average reward (last 10 episodes)
  ✓ Max/min reward in current batch
  ✓ Episode length (steps per episode)
  ✓ Success rate (% reaching goal)
  ✓ All loss components
  ✓ Cumulative steps


# ============================================================================
# ANALYSIS TOOL - AUTOMATED REPORTING & VISUALIZATION
# ============================================================================

LOCATION:
  analyze_training.py (in experiments/Cole directory)

FEATURES:
  • Loads CSV metrics and generates plots
  • Calculates training statistics
  • Exports human-readable report
  • Detects performance trends
  • Creates multi-panel figures

USAGE AFTER TRAINING:

1. PRINT SUMMARY STATISTICS:
   python analyze_training.py --log-dir ./runs/spot_rl
   
   Output:
   ================================================================================
   TRAINING SUMMARY STATISTICS
   ================================================================================
   Episodes Trained: 1000
   Total Steps:      425,000
   
   --- REWARD METRICS ---
     Max Reward:      234.5678
     Min Reward:       12.3456
     Final Reward:    198.4567
     Mean Reward:     145.2343
     Reward Trend:    ↑ IMPROVING (18.3%)
   ...

2. GENERATE PLOTS & REPORT:
   python analyze_training.py --log-dir ./runs/spot_rl --full
   
   Creates:
   - training_curves.png  (6-panel visualization)
   - analysis_report.txt  (detailed statistics)

3. CUSTOM OUTPUT PATHS:
   python analyze_training.py \
     --csv ./runs/spot_rl/training_metrics.csv \
     --plot ./my_training_plot.png \
     --report ./my_training_analysis.txt

PLOT PANELS (6 total):
  [Top Left]     Episode Reward Trajectory (with uncertainty band)
  [Top Center]   Success Rate Over Time
  [Top Right]    Training Losses (Policy/Value/Total on log scale)
  [Bottom Left]  Episode Length Distribution
  [Bottom Center] Policy Entropy (Exploration Behavior)
  [Bottom Right]  Reward Distribution by Training Phase

REPORT INCLUDES:
  • Summary statistics (episodes, steps)
  • Reward performance metrics
  • Success rate analysis
  • Loss convergence analysis
  • Exploration behavior
  • Episode timing statistics
  • Training efficiency estimate
  • Trend analysis {IMPROVING/DECLINING/STABLE}


# ============================================================================
# QUICK START - ALL TRACKING AT ONCE
# ============================================================================

TERMINAL WINDOW 1 - Start Training:
  cd c:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole
  C:\isaac-sim\python.bat SpotRL_Training.py --episodes 1000

  [Wait for this message]
  ================================================================================
  TRAINING TRACKING SETUP
  ================================================================================
  TensorBoard logs:  ./runs/spot_rl
  CSV logs:          ./runs/spot_rl/training_metrics.csv
  Checkpoints:       ./checkpoints/spot_rl
  
  To monitor training in real-time, run:
    tensorboard --logdir ./runs/spot_rl --port 6006
  Then open http://localhost:6006 in your browser
  ================================================================================

TERMINAL WINDOW 2 - Real-Time Monitoring:
  cd c:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole
  tensorboard --logdir ./runs/spot_rl --port 6006

  [Copy URL to browser]
  http://localhost:6006

AFTER TRAINING - Analysis:
  python analyze_training.py --log-dir ./runs/spot_rl --full

  [Opens plots and generates report automatically]


# ============================================================================
# INTERPRETING TRAINING METRICS
# ============================================================================

✓ REWARD INCREASING:
  Sign of good learning! Policy discovering better navigation strategies.
  Goal: Monotonic increase without plateaus.

✓ SUCCESS RATE RISING:
  More episodes reaching the goal.
  Goal: Reach 80%+ success rate.

✓ POLICY LOSS DECREASING:
  Policy gradient refinement is effective.
  Goal: Smooth decrease toward stable value (~0.05-0.1).

✓ VALUE LOSS DECREASING:
  Value function estimates becoming more accurate.
  Goal: Decrease toward near-zero and stabilize.

✓ ENTROPY DECREASING:
  Policy becoming more deterministic (good - learned strategy).
  Goal: Gradual decrease from ~high to ~low.
  Avoid: Sudden entropy drops (may indicate premature convergence).

⚠ REWARD PLATEAU:
  Policy learned but not improving further.
  Solutions: Increase learning rate, add curriculum, adjust rewards.

⚠ LOSS DIVERGING (Rising):
  Unstable training, likely learning rate too high.
  Solutions: Reduce learning_rate in config, restart from checkpoint.

⚠ ENTROPY TOO HIGH:
  Policy still random, not converging.
  Solutions: Reduce entropy_coef in config, increase training episodes.

⚠ SUCCESS RATE STAYING LOW:
  Robot not reaching goal consistently.
  Solutions: Check reward shaping, verify motor control, adjust episode length.


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

Q: TensorBoard shows no data
A: Ensure training has created ./runs/spot_rl/events.* files
   Check that tensorboard points to correct --logdir path

Q: CSV file not updating
A: May be buffered. Stop training and check file.
   Training script flushes on every logging interval.

Q: Can't connect to TensorBoard http://localhost:6006
A: Port 6006 already in use?
   Try: tensorboard --logdir ./runs/spot_rl --port 6007
   Or kill other tensorboard:
   taskkill /IM tensorflow.exe /F

Q: Plots look wrong in analyze_training.py
A: Ensure pandas/matplotlib installed:
   pip install pandas matplotlib
   
Q: CSV has missing rows
A: May still be writing. Train longer. Final checkpoint will flush all.

Q: Want to compare multiple training runs
A: Run multiple training sessions with different names:
   python analyze_training.py --log-dir ./runs/spot_rl_v1
   python analyze_training.py --log-dir ./runs/spot_rl_v2
   
   Manual comparison:
   import pandas as pd
   df1 = pd.read_csv('./runs/spot_rl_v1/training_metrics.csv')
   df2 = pd.read_csv('./runs/spot_rl_v2/training_metrics.csv')
   
   # Plot both
   df1['avg_reward'].plot(label='V1')
   df2['avg_reward'].plot(label='V2')


# ============================================================================
# FILE STRUCTURE AFTER TRAINING
# ============================================================================

After running training for N episodes, you'll have:

runs/
  spot_rl/
    events.out.tfevents.X........  (TensorBoard data files)
    training_metrics.csv         ← CSV logs (main analysis file)
    training_curves.png          ← Plot (if ran analyze_training.py)
    analysis_report.txt          ← Report (if ran analyze_training.py)

checkpoints/
  spot_rl/
    spot_rl_ep50.pt             ← Network weights + optimizer
    spot_rl_ep50.json           ← Metadata (readable JSON)
    spot_rl_ep100.pt
    spot_rl_ep100.json
    ...
    spot_rl_final.pt            ← Final model
    spot_rl_final.json

Environment2_flat_terrain.py
SpotRL_Environment.py
SpotRL_Training.py
analyze_training.py
...


# ============================================================================
# NEXT STEPS AFTER TRAINING
# ============================================================================

1. ANALYZE RESULTS:
   python analyze_training.py --log-dir ./runs/spot_rl --full

2. EVALUATE BEST CHECKPOINT:
   # Code to load best model and run evaluation
   checkpoint = torch.load('./checkpoints/spot_rl/spot_rl_ep1000.pt')

3. VISUALIZE POLICY BEHAVIOR:
   # Record video of trained agent navigating

4. TRANSFER TO REAL ROBOT:
   # Export weights to real Spot hardware

5. FINE-TUNE ON REAL TERRAIN:
   # Curriculum learning with more complex environments

"
