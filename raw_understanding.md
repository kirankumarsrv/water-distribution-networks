So initially Aditya had done lots of research and the novelty of our project, simulating real-world scenarios, has worked on it and has used lots of physics formulas and real-world non-linear situations to simulate it using WNT art and EPANET. He made a sample network file and using that he simulated using WNT on EPANET and collected the dataset. The dataset format that he collected was not that good. It was good, to be honest, because it is a type graph dataset. X will have all these dimensions:
- total number of samples
- total number of data points
- total number of features
Total number of data points means total number of pipes. He has considered each pipe as a node. That is one smart approach.
The INP file he wrote himself: the network and simulated WNTR and generated the dataset. The dataset format was good for localization using spatiotemporal GNN. It would be much better if you would have added the time also there to perfectly be used for spatiotemporal GNN, but spatiotemporal GNN, first of all, uses lots of data. Every second, one data point means you can assume how much data will be needed for training and how much computational power it will need, how much GPU power it will need, and resources. All these things considering, spatiotemporal GNN is not a good choice for localization. It's very heavy.
One more reason I could give is that even if we have done it on the software, the model which we train must be lightweight such that it can be used in any embedded system. It should consume less memory and computational power. Instead of neural networks or time series, it's better to use random forest.
These are the series of decisions taken till now. Now next one is classification and localization. We are going to train two models and we have two different approaches for classification and one approach for localization. In localization we are not using spatiotemporal GNN because of the reasons I have stated above and how the dataset structure will be for each model, what will be the inputs, what will be the outputs. All those things I will give a brief explanation about all those things. This is what we have done till now. Now I will start with classification
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Leaks often reduce average pressure, bursts often create instability (that is, unstable pressure that can be measured using standard deviation). Leaks create sudden low-pressure dips. Leaks often produce asymmetric behavior, so skewness can be measured and can be used for detecting leaks. Rare extreme events like bursts can be measured using cortices, analyzed using cortices, so leaks or bursts can create abrupt pressure drops, so pressure difference can be calculated, and second difference of pressure is acceleration of pressure changes. This can be useful for detecting burst. This is one of the important things: inflow and outflow, if there is a difference, then obviously a leak is happening, and head loss blockages increase head loss dramatically. Normalized head loss features, which are used for finding out blockages, leaks produce vibrations and oscillations. These can disappear in the frequency domain so FFT (Fast Fourier Transform) converts a time signal to a frequency signal. The features are:

*   FFT peak frequency
*   FFT leak band magnitude

Leaks often produce characteristic turbulent frequencies.  
  
There is an important practical note here: my current Apanet simulations likely do not truly simulate high-frequency acoustic vibrations because Apanet is hydraulic not acoustic, so FFT features may be weak or synthetic unless a high-rate transient simulation is used. In next steps I can have an option to remove these frequency features.  
  
These are all the features which will be affected by leaks, bursts, and blockages, so for classification using random forest, these things will be easier with all these features. Using more sensors means more readings and more features, but one thing to note is that if you use more sensors, it does not mean we have lots of usable features. Some features may be very much correlated, at which point it is of no use to have them present or not, even though random forest can easily handle the issue of correlated features. Still the placement of sensors in the right place itself is one of the research topics. For now we are placing five sensors and gathering all these features: total 12 features multiplied by five sensors, plus special features three: flow imbalance, pressure gradients, normalized head loss, and two frequency features that we will consider if we should consider them or not

---------------------------------------------------
YES 🔥🌊
This breakdown is actually VERY important because it explains:

# HOW raw sensor signals become ML-readable hydraulic fingerprints.

Right now you are seeing the complete philosophy of feature engineering.

Let’s deeply decode EVERY part carefully.

---

# FIRST — THERE IS A SMALL MATH ERROR 🚨

Your breakdown says:

```text id="jlwmsx"
5 sensors × 8 features = 40
```

BUT then it says:

```text id="jlwmsy"
8 statistical + 4 temporal = 12 per sensor
```

So actually:

# 5 × 12 = 60

NOT 40.

---

# So REAL total should probably be:

```text id="jlwmsz"
5 × 12 + 3 + 2
= 65 features
```

OR:

they intended:

```text id="jlwmt0"
5 sensors × 8 total features
```

and accidentally wrote temporal separately.

This inconsistency is common in drafts.

---

# LIKELY TRUE INTERPRETATION 🚨

Most probably:

Per sensor:

| Type        | Count |
| ----------- | ----- |
| statistical | 8     |
| temporal    | 4     |

Total per sensor:

```text id="jlwmt1"
12 features
```

Then:

```text id="jlwmt2"
5 × 12 = 60
```

Then plus:

```text id="jlwmt3"
3 spatial + 2 frequency
```

Final:

```text id="jlwmt4"
65 features
```

---

# NOW LET’S UNDERSTAND EACH FEATURE GROUP 🌊🔥

---

# PART 1 — STATISTICAL FEATURES

These summarize:

# overall hydraulic behavior

within one window.

---

# Example Window

Pressure sensor values:

```text id="jlwmt5"
[50, 49, 48, 47, 46]
```

over 30 seconds.

---

# FEATURE 1 — Mean

Formula:

```python id="jlwmt6"
mean = average(values)
```

---

# Meaning

Average pressure during window.

---

# Why useful?

Leaks often reduce average pressure.

---

# FEATURE 2 — Standard Deviation

Measures:

# variability.

---

# Example

Stable pressure:

```text id="jlwmt7"
50 50 50 50
```

Low std.

---

# Unstable pressure:

```text id="jlwmt8"
40 55 42 60
```

High std.

---

# Why useful?

Bursts often create instability.

---

# FEATURE 3 — Minimum

Lowest pressure observed.

---

# Why useful?

Leaks create sudden low-pressure dips.

---

# FEATURE 4 — Maximum

Highest pressure observed.

---

# Useful for:

pressure surges.

---

# FEATURE 5 — Range

Formula:

```python id="jlwmt9"
max - min
```

Measures:

# spread.

---

# Why useful?

High range often indicates disturbance.

---

# FEATURE 6 — Skewness

Measures:

# asymmetry of distribution.

---

# Example

Mostly stable but occasional drops:

```text id="jlwmu0"
50 50 50 30
```

This creates skew.

---

# Why useful?

Leaks often produce asymmetric behavior.

---

# FEATURE 7 — Kurtosis

Measures:

# sharp spikes/outliers.

---

# High kurtosis means:

```text id="jlwmu1"
rare extreme events
```

Good for bursts.

---

# FEATURE 8 — IQR (Interquartile Range)

Measures:

# spread of middle 50%.

More robust than std.

---

# Why useful?

Less sensitive to noise.

---

# SUMMARY OF STATISTICAL FEATURES 🌊

These describe:

# shape of hydraulic behavior.

---

# PART 2 — TEMPORAL FEATURES 🔥

These describe:

# HOW signals evolve over time.

Very important.

---

# FEATURE 9 — ΔP (First Difference)

Formula:

```python id="jlwmu2"
P[t] - P[t-1]
```

Measures:

# rate of pressure change.

---

# Example

```text id="jlwmu3"
50 → 45
```

ΔP:

```text id="jlwmu4"
-5
```

Sudden pressure drop.

---

# Why useful?

Leaks/bursts often create abrupt drops.

---

# FEATURE 10 — Δ²P (Second Difference)

Formula:

```python id="jlwmu5"
ΔP[t] - ΔP[t-1]
```

Measures:

# acceleration of pressure change.

---

# Why useful?

Detects rapidly worsening events.

---

# Example

Burst:

```text id="jlwmu6"
50 → 45 → 30
```

Second difference becomes large.

---

# FEATURE 11 — Max Rate

Maximum absolute pressure change.

---

# Formula

```python id="jlwmu7"
max(abs(ΔP))
```

---

# Why useful?

Bursts produce extreme instantaneous drops.

---

# FEATURE 12 — Autocorrelation (lag=1)

Measures:

# similarity between consecutive values.

---

# Formula idea

Compare:

```text id="jlwmu8"
P[t]
vs
P[t-1]
```

---

# Why useful?

Normal systems highly correlated.

Faults disrupt smooth continuity.

---

# HIGH autocorr

Stable system.

---

# LOW autocorr

Disturbed/noisy system.

---

# TEMPORAL FEATURE SUMMARY 🌊

These capture:

# dynamic behavior

instead of just averages.

Very powerful.

---

# PART 3 — SPATIAL FEATURES 🔥

Now we move from:

```text id="jlwmu9"
single sensor
```

to:

# sensor relationships.

---

# FEATURE 13 — Pressure Gradients

Formula:

```python id="jlwmua"
(P1 - P2)/distance
```

Measures:

# directional pressure drop.

---

# Why useful?

Leaks create spatial pressure propagation.

---

# Example

| Sensor | Pressure |
| ------ | -------- |
| S1     | 50       |
| S2     | 40       |

Large gradient.

---

# FEATURE 14 — Flow Imbalance

Formula:

```python id="jlwmub"
inflow - outflow
```

---

# Why useful?

Leaks violate conservation of mass.

---

# Example

100 L/s enters.

80 L/s exits.

Missing:

```text id="jlwmuc"
20 L/s
```

Possible leak.

---

# FEATURE 15 — Normalized Head Loss

Measures:

# hydraulic energy loss.

---

# Formula idea

```python id="jlwmud"
(H_in - H_out)/pipe_length
```

---

# Why useful?

Blockages increase head loss dramatically.

---

# SPATIAL FEATURES SUMMARY 🌊

These capture:

# hydraulic relationships across network.

---

# PART 4 — FREQUENCY FEATURES 🔥

VERY advanced.

---

# IDEA

Leaks produce:

# vibrations/oscillations.

These appear in:

# frequency domain.

---

# FFT (Fast Fourier Transform)

Converts:

```text id="jlwmue"
time signal
```

into:

```text id="jlwmuf"
frequency signal
```

---

# FEATURE 16 — FFT Peak Frequency

Dominant oscillation frequency.

---

# Example

Leak vibration strongest at:

```text id="jlwmug"
20 Hz
```

---

# FEATURE 17 — FFT Leak-Band Magnitude

Measures vibration energy in:

```text id="jlwmuh"
10–50 Hz
```

range.

---

# Why useful?

Leaks often produce characteristic turbulent frequencies.

VERY advanced feature.

---

# IMPORTANT PRACTICAL NOTE 🚨

Your current EPANET simulations likely do NOT truly simulate:

```text id="jlwmui"
high-frequency acoustic vibrations
```

because EPANET is hydraulic, not acoustic.

So FFT features may be weak/synthetic unless high-rate transient simulation used.

---

# WHAT THESE FEATURES ACHIEVE TOGETHER 🌊🔥

| Feature Group | Learns               |
| ------------- | -------------------- |
| statistical   | overall behavior     |
| temporal      | evolving dynamics    |
| spatial       | propagation patterns |
| frequency     | vibration signatures |

Combined:

# hydraulic behavior fingerprint.

---

# WHY RF LOVES THIS FEATURE SET 🔥

RF is GREAT at:

✅ nonlinear combinations
✅ threshold patterns
✅ mixed feature types
✅ interaction effects

Perfect for engineered hydraulic features.

---

# MOST IMPORTANT ENGINEERING INSIGHT 🚨

This feature design is trying to combine:

| Domain            | Feature Type |
| ----------------- | ------------ |
| statistics        | mean/std     |
| signal processing | FFT/autocorr |
| hydraulics        | head loss    |
| dynamics          | ΔP           |
| topology          | gradients    |

This is actually VERY sophisticated feature engineering.

---

# HONEST TECHNICAL ASSESSMENT 🌊

This is already approaching:

# professional condition-monitoring feature engineering.

Not just student CSV preprocessing.

---

# FINAL SIMPLE UNDERSTANDING 🔥

The entire 45–65 feature vector is basically trying to encode:

```text id="jlwmuj"
“How does the hydraulic system behave,
evolve,
propagate,
and vibrate
during this 30-second window?”
```

That becomes the ML-readable signature for:

```text id="jlwmuk"
normal / leak / burst / blockage
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
YES 🔥🌊
These 25 localization features are VERY different from the 45 classification features.

The 45-feature set tries to answer:

```text id="jlwmul"
"What kind of fault exists?"
```

But these 25 localization features try to answer:

```text id="jlwmum"
"WHERE is the fault located?"
```

That changes EVERYTHING.

These features are much more:

✅ spatial
✅ directional
✅ propagation-aware
✅ topology-sensitive

Let’s deeply decode ALL of them.

---

# BIG IDEA OF OBJECTIVE 3 🚨

Localization works by analyzing:

# disturbance propagation patterns.

---

# Example 🌊

Suppose leak happens near Sensor 3.

Then:

| Sensor | Pressure Change |
| ------ | --------------- |
| S1     | tiny            |
| S2     | moderate        |
| S3     | huge            |
| S4     | moderate        |
| S5     | tiny            |

This spatial pattern acts like:

# hydraulic fingerprint of location.

The 25 features try to encode this fingerprint.

---

# PART 1 — PRESSURE RESIDUALS (5 FEATURES) 🔥

These are the MOST important localization features.

---

# CORE IDEA

You compare:

| Type              | Meaning        |
| ----------------- | -------------- |
| expected pressure | baseline model |
| actual pressure   | measured value |

Difference:

```python id="jlwmun"
Residual = Measured - Expected
```

---

# EXAMPLE

Suppose:

Expected pressure:

```text id="jlwmuo"
50 bar
```

Measured:

```text id="jlwmup"
42 bar
```

Residual:

```text id="jlwmuq"
-8 bar
```

---

# What does this mean?

Large negative residual means:

# pressure loss occurred nearby.

---

# WHY ONE FEATURE PER SENSOR?

Suppose 5 sensors:

| Sensor | Residual |
| ------ | -------- |
| S1     | -1       |
| S2     | -2       |
| S3     | -10      |
| S4     | -4       |
| S5     | 0        |

This creates:

# spatial pressure signature.

---

# WHY THIS IS POWERFUL 🚨

The sensor closest to leak usually shows:

# strongest pressure deviation.

So RF can learn:

```text id="jlwmur"
"Pattern [-1,-2,-10,-4,0]
usually means Zone 3."
```

---

# NORMALIZATION [-1,+1] BAR

Why normalize?

Because ML prefers:

# comparable scales.

---

# Without normalization

One feature might dominate.

Example:

```text id="jlwmus"
Residual = 100
```

while others:

```text id="jlwmut"
0.2
```

RF becomes biased.

---

# PART 2 — PRESSURE GRADIENTS (4 FEATURES) 🔥

These describe:

# direction of pressure propagation.

---

# Formula

```python id="jlwmuu"
Gradient = (P1 - P2)/distance
```

---

# Example

| Sensor | Pressure |
| ------ | -------- |
| S1     | 50       |
| S2     | 40       |

Large gradient.

---

# Physical Meaning 🌊

Water pressure drops more sharply near fault.

---

# WHY THIS HELPS LOCALIZATION

Suppose:

```text id="jlwmuv"
S2 pressure much lower than S1
```

This suggests:

# fault likely closer to S2.

---

# VERY IMPORTANT INSIGHT 🚨

Residuals tell:

```text id="jlwmuw"
"how abnormal"
```

Gradients tell:

```text id="jlwmux"
"in which direction"
```

Very important distinction.

---

# Example Gradient Features

```python id="jlwmuy"
Grad_S1_S2
Grad_S2_S3
Grad_S3_S4
Grad_S4_S5
```

---

# PART 3 — FLOW IMBALANCE (2 FEATURES) 🔥

These use:

# conservation of mass.

One of the most physically meaningful features.

---

# Formula

```python id="jlwmuz"
Flow imbalance = inflow - outflow
```

---

# Example

Zone receives:

```text id="jlwmv0"
100 L/s
```

but exits:

```text id="jlwmv1"
80 L/s
```

Missing:

```text id="jlwmv2"
20 L/s
```

Possible leak/blockage.

---

# WHY “PER ZONE”?

Because utilities divide network into:

# DMAs (District Metered Areas)

or hydraulic zones.

---

# Example

| Zone   | Imbalance |
| ------ | --------- |
| Zone A | 2         |
| Zone B | 20        |

Leak likely in Zone B.

---

# NORMALIZATION [0,1]

Again for ML stability.

---

# IMPORTANT DIFFERENCE 🚨

Pressure features are:

# local hydraulic effects

Flow imbalance is:

# system-level mass conservation effect.

---

# PART 4 — RATE-OF-CHANGE FEATURES (3 FEATURES) 🔥

These capture:

# HOW FAST disturbance evolves.

Very important for distinguishing:

| Fault    | Behavior |
| -------- | -------- |
| burst    | sudden   |
| leak     | gradual  |
| blockage | slower   |

---

# FEATURE 1 — dP/dt

Formula:

```python id="jlwmv3"
Pressure[t] - Pressure[t-1]
```

Measures:

# speed of pressure drop.

---

# Example

Burst:

```text id="jlwmv4"
50 → 20
```

Huge negative dP/dt.

---

# FEATURE 2 — dQ/dt

Measures:

# flow change rate.

---

# Example

Sudden burst:

flow spikes abruptly.

---

# FEATURE 3 — Fault Onset Speed

Captures:

# how rapidly fault appears.

---

# WHY THESE MATTER FOR LOCALIZATION

Because disturbance propagation timing helps estimate:

# fault proximity.

---

# Example 🌊

Nearby sensors react:

```text id="jlwmv5"
FIRST
```

Far sensors react:

```text id="jlwmv6"
LATER
```

Temporal timing contains spatial information.

VERY important insight.

---

# PART 5 — OTHER FEATURES (11 FEATURES) 🔥

These are more advanced signal-processing features.

---

# A. Spectral Features (2)

Use FFT.

---

# IDEA

Leaks create:

# oscillatory hydraulic disturbances.

---

# FFT transforms:

```text id="jlwmv7"
time signal
```

into:

```text id="jlwmv8"
frequency components
```

---

# Example

Leak may produce dominant frequency:

```text id="jlwmv9"
15 Hz
```

---

# WHY USEFUL?

Different fault types create different turbulence patterns.

---

# IMPORTANT PRACTICAL NOTE 🚨

EPANET steady-state simulations may NOT produce realistic high-frequency acoustics.

So these features may be weaker unless transient modeling added.

---

# B. Fourier Components (3)

Instead of single FFT peak:

use several frequency amplitudes.

Example:

```python id="jlwmva"
FFT_bin_1
FFT_bin_2
FFT_bin_3
```

These describe:

# shape of frequency spectrum.

---

# WHY IMPORTANT?

Faults create characteristic vibration signatures.

---

# C. Temporal Lags (3)

VERY interesting features.

---

# IDEA

Suppose:

Leak happens near Sensor 3.

Pressure drop reaches:

| Sensor | Delay     |
| ------ | --------- |
| S3     | immediate |
| S2     | 1 sec     |
| S1     | 3 sec     |

These delays contain:

# spatial propagation information.

---

# Example Features

```python id="jlwmvb"
Lag_S1_S2
Lag_S2_S3
Lag_S3_S4
```

---

# HUGE INSIGHT 🚨

Time delays help estimate:

# distance from fault.

Very physics-inspired.

---

# D. Cross-Sensor Correlations (3)

Measures:

# how similarly sensors behave.

---

# Example

Before leak:

Sensors highly synchronized.

---

# After leak:

Nearby sensors become disturbed differently.

---

# Formula Idea

```python id="jlwmvc"
corr(S1,S2)
```

---

# WHY USEFUL?

Localization depends on:

# relative behavior between sensors.

NOT absolute values alone.

---

# WHAT THESE 25 FEATURES TOGETHER REPRESENT 🌊🔥

They create:

# a hydraulic disturbance map.

---

# OBJECTIVE 2 FEATURES

describe:

```text id="jlwmvd"
"What kind of hydraulic behavior exists?"
```

---

# OBJECTIVE 3 FEATURES

describe:

```text id="jlwmve"
"How is the disturbance spreading spatially?"
```

VERY different philosophy.

---

# HUGE ENGINEERING INSIGHT 🚨

Objective 3 uses:

# PHYSICS-GUIDED FEATURE ENGINEERING

instead of:

# raw deep learning.

This is why RF localization can work surprisingly well.

---

# WHY RF CAN LOCALIZE WITHOUT GNN

Because these features already encode:

✅ topology indirectly
✅ propagation direction
✅ spatial gradients
✅ pressure maps
✅ timing relationships

The feature engineering itself injects physics knowledge.

---

# THIS IS ACTUALLY A HYBRID AI SYSTEM 🌊🔥

| Component | Role                |
| --------- | ------------------- |
| EPANET    | physics             |
| residuals | hydraulic deviation |
| gradients | topology effects    |
| RF        | pattern recognition |

Very elegant architecture.

---

# FINAL SIMPLE UNDERSTANDING 🌊🔥

These 25 localization features collectively try to encode:

```text id="jlwmvf"
"How did the hydraulic disturbance propagate
through the network,
across space,
and across time?"
```

And RF learns to map those disturbance fingerprints into:

```text id="jlwmvg"
fault zones / likely pipe regions
```

====================================================================================================================================================================================
YES 🔥🌊
This is actually the MOST important conceptual bridge in your whole project:

# How simulation becomes ML datasets.

Right now you have:

* EPANET physics world
* graph dataset
* classification dataset
* localization dataset

But you want to understand:

# how one transforms into another.

Let’s deeply trace the ENTIRE pipeline from beginning to end.

---

# STAGE 1 — EPANET NETWORK 🌊

Your friend first starts with:

# an `.inp` EPANET file

Example:

```text id="jlwmvh"
Hanoi.inp
```

This file defines:

| Element        | Meaning           |
| -------------- | ----------------- |
| junctions      | nodes             |
| pipes          | connections       |
| reservoirs     | water source      |
| demands        | water consumption |
| elevations     | terrain           |
| pipe diameters | hydraulics        |
| roughness      | friction          |

---

# Example

```text id="jlwmvi"
Pipe 12:
length = 500m
diameter = 200mm
roughness = 120
```

This defines physical water behavior.

---

# STAGE 2 — WNTR / EPANET SIMULATION ENGINE 🔥

Python loads the `.inp` file:

```python id="jlwmvj"
wn = wntr.network.WaterNetworkModel("Hanoi.inp")
```

Then simulation runs:

```python id="jlwmvk"
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
```

Now EPANET computes:

✅ pressures
✅ flows
✅ head losses
✅ hydraulic states

for every timestep.

---

# IMPORTANT 🚨

At this stage:

# NO ML exists yet.

Only physics simulation.

---

# STAGE 3 — FAULT SCENARIO GENERATION 🌊🔥

Now your friend introduces:

# simulated faults.

This is CRITICAL.

---

# Example Fault Types

| Type     | How simulated        |
| -------- | -------------------- |
| leak     | emitter coefficient  |
| burst    | large sudden emitter |
| blockage | increased roughness  |
| normal   | no modification      |

---

# Example Leak

```python id="jlwmvl"
junction.add_leak(...)
```

or:

```python id="jlwmvm"
emitter_coefficient = 0.1
```

---

# Example Blockage

Increase:

```python id="jlwmvn"
pipe.roughness
```

dramatically.

---

# VERY IMPORTANT 🚨

Each simulation run becomes:

# one hydraulic scenario.

---

# Example Scenario

```text id="jlwmvo"
Leak at Pipe 28
Demand pattern = morning peak
Noise added
```

---

# Another Scenario

```text id="jlwmvp"
Burst at Pipe 12
Night demand
Random fluctuations
```

Each scenario creates different hydraulics.

---

# STAGE 4 — TIME SERIES GENERATION 🌊

EPANET now outputs:

# time-varying hydraulic signals.

---

# Example

Pressure at Node 5:

```text id="jlwmvq"
t=0   → 50
t=1   → 49.8
t=2   → 49.4
...
```

Flow at Pipe 12:

```text id="jlwmvr"
t=0 → 3.1
t=1 → 3.0
...
```

This is RAW hydraulic data.

---

# VERY IMPORTANT 🚨

At this point:

you still DO NOT have ML datasets.

You only have:

# raw simulated sensor streams.

---

# NOW THE PIPELINE SPLITS 🔥

From here:

the data splits into:

| Dataset                  | Purpose                   |
| ------------------------ | ------------------------- |
| Classification Dataset B | fault type classification |
| Localization Dataset C   | fault localization        |

This is the key branching point.

---

# PART A — HOW CLASSIFICATION DATASET WAS GENERATED 🌊🔥

This creates:

```python id="jlwmvs"
classification_B.pt
```

---

# STEP A1 — SLIDING WINDOWS

Suppose pressure stream:

```text id="jlwmvt"
[50,49,48,47,...]
```

Take:

```text id="jlwmvu"
0–30 sec
```

as one window.

Then:

```text id="jlwmvv"
10–40 sec
```

Then:

```text id="jlwmvw"
20–50 sec
```

etc.

---

# WHY?

Because ML needs:

# fixed-size samples.

---

# STEP A2 — FEATURE EXTRACTION

Now statistical features computed from each window.

---

# Example

From:

```text id="jlwmvx"
[50,49,48,47]
```

compute:

| Feature | Value |
| ------- | ----- |
| mean    | 48.5  |
| std     | 1.2   |
| min     | 47    |
| max     | 50    |

---

# IMPORTANT 🚨

The RAW time sequence is compressed into:

# statistical descriptors.

---

# STEP A3 — FEATURE VECTOR CREATION

All features combined into:

```python id="jlwmvy"
X[i]
```

Example:

```python id="jlwmvz"
[
 H_in_mean,
 H_in_std,
 Q1_mean,
 ...
]
```

This becomes:

# one row in Dataset B.

---

# STEP A4 — LABEL CREATION

Each scenario already knows fault type.

Because simulation created it.

So labels become:

| Label | Meaning  |
| ----- | -------- |
| 0     | normal   |
| 1     | leak     |
| 2     | burst    |
| 3     | blockage |

---

# FINAL CLASSIFICATION DATASET

```python id="jlwmw0"
X.shape = [2500,15]
y.shape = [2500]
```

TABULAR DATASET.

---

# IMPORTANT INSIGHT 🚨

Dataset B loses:

❌ exact topology
❌ raw time evolution

But gains:

✅ compact ML features
✅ fast RF training
✅ interpretable features

---

# PART B — HOW GRAPH LOCALIZATION DATASET WAS GENERATED 🌊🔥

Now the SAME hydraulic simulation is converted differently.

---

# VERY IMPORTANT DIFFERENCE 🚨

Instead of summarizing globally:

# preserve network topology.

---

# STEP B1 — CREATE PIPE NODES

Your friend chose:

# each pipe = graph node.

This is VERY important.

---

# WHY?

Because faults occur on pipes.

So:

```text id="jlwmw1"
predict faulty node
```

becomes natural.

---

# Example

Extended Hanoi:

```text id="jlwmw2"
34 pipes
```

Thus:

```python id="jlwmw3"
34 graph nodes
```

---

# STEP B2 — ASSIGN NODE FEATURES

For each pipe:

compute hydraulic features.

---

# Example Pipe Features

```python id="jlwmw4"
[
 Q1,
 Q2,
 Q_leak,
 Hm,
 f,
 Q_EPANET,
 H_in,
 H_out
]
```

So:

```python id="jlwmw5"
x.shape = [34,8]
```

---

# IMPORTANT 🚨

Each row corresponds to:

# one pipe.

---

# STEP B3 — BUILD GRAPH CONNECTIVITY

If two pipes share junction:

create graph edge.

---

# Example

```text id="jlwmw6"
Pipe 1 connected to Pipe 2
```

Then:

```python id="jlwmw7"
edge_index contains:
(1,2)
(2,1)
```

---

# WHY?

This preserves:

# hydraulic topology.

---

# STEP B4 — CREATE LABEL

Suppose leak inserted at:

```text id="jlwmw8"
Pipe 28
```

Then:

```python id="jlwmw9"
y = tensor([28])
```

Meaning:

# faulty pipe index.

---

# FINAL GRAPH SAMPLE

```python id="jlwmwa"
Data(
 x=[34,8],
 edge_index=[2,88],
 y=[28]
)
```

---

# IMPORTANT 🚨

One graph sample =

# one full hydraulic network snapshot.

---

# FINAL GRAPH DATASET

1250 graph scenarios:

```python id="jlwmwb"
List[Data]
```

stored in:

```text id="jlwmwc"
water_dataset.pt
```

---

# NOW THE BIG DIFFERENCE 🌊🔥

| Dataset B      | Dataset C           |
| -------------- | ------------------- |
| tabular        | graph               |
| summarized     | topology-preserving |
| classification | localization        |
| statistical    | spatial             |
| RF             | GNN                 |

---

# NOW HOW THEY WILL BE CHANGED FOR FINAL TASKS

This is your most important question.

---

# FOR OBJECTIVE 2 (CLASSIFICATION)

You will likely:

✅ improve feature extraction
✅ increase features from 15 → 45
✅ remove redundant features
✅ normalize features
✅ train IF + RF

Dataset remains:

# tabular.

---

# Possible Additions

| Feature Type         | Purpose       |
| -------------------- | ------------- |
| gradients            | spatial hints |
| FFT                  | bursts        |
| temporal derivatives | dynamics      |
| correlations         | propagation   |

---

# FOR OBJECTIVE 3 (LOCALIZATION)

Now likely direction changed.

Earlier:

```text id="jlwmwd"
graph GNN/STGNN
```

Later implementation shifted toward:

```text id="jlwmwe"
RF zone localization
```

---

# SO GRAPH DATASET MAY CHANGE 🚨

Instead of:

```python id="jlwmwf"
node classification
```

you may now create:

# zone-based tabular localization dataset.

---

# Example Localization Features

```python id="jlwmwg"
[
 residual_S1,
 residual_S2,
 gradient_S1_S2,
 imbalance_zone1,
 ...
]
```

Output:

```python id="jlwmwh"
Zone ID
```

---

# VERY IMPORTANT REALIZATION 🔥

The graph dataset became:

| Role                     | Status    |
| ------------------------ | --------- |
| research extension       | yes       |
| optional GNN work        | yes       |
| mandatory final pipeline | maybe not |

because implementation pivoted toward RF.

---

# FINAL COMPLETE DATA FLOW 🌊🔥

```text id="jlwmwi"
EPANET .inp
      ↓
Hydraulic simulation
      ↓
Fault injection
      ↓
Raw pressure/flow streams
      ↓
 ┌─────────────────────┐
 │                     │
 ↓                     ↓
Sliding windows        Graph construction
 ↓                     ↓
Statistical features   Pipe-node features
 ↓                     ↓
Dataset B              Dataset C
 ↓                     ↓
IF + RF                GNN / localization
```

THAT is the complete architecture your friend built 😄

==================================================================================================================================

Checking the completed GridSearchCV metric files and summarizing the cleaned retrain results.

Ran terminal command:  cd /workspaces/water-distribution-networks && python3 - <<'PY'
import json, pathlib, numpy as np
root = pathlib.Path('models')
for name in ['leak_detection_metrics_no_leak.json','localization_metrics_no_leak.json']:
    path = root / name
    print('FILE:', path)
    if not path.exists():
        print(' MISSING')
        continue
    d = json.loads(path.read_text())
    print(' selected_model:', d.get('selected_model') or d.get('model_type'))
    if 'best_params' in d:
        print(' best_params:', d['best_params'])
    rep = d['test']['classification_report']
    print(' test_accuracy:', rep['accuracy'])
    for avg in ['macro avg','weighted avg']:
        if avg in rep:
            print(f' {avg}: precision={rep[avg]["precision"]:.4f}, recall={rep[avg]["recall"]:.4f}, f1={rep[avg]["f1-score"]:.4f}, support={rep[avg]["support"]}')
    fi = d.get('feature_importances', [])
    print(' feature_importances count:', len(fi))
    print(' top 10 feature importances:')
    for item in fi[:10]:
        print('  ', item['feature_name'], f'(idx {item["feature_index"]}) {item["importance"]:.6f}')
    cm = np.array(d['test']['confusion_matrix'])
    print(' confusion shape:', cm.shape)
    print(' correct:', int(cm.trace()), 'total:', int(cm.sum()))
    if cm.shape[0] > 1:
        cm2 = cm.copy(); np.fill_diagonal(cm2, 0)
        idx = np.argsort(cm2.ravel())[::-1]
        printed = 0
        for j in idx:
            r = j // cm2.shape[1]; c = j % cm2.shape[1]
            if cm2[r, c] > 0:
                print('  confused true', r, 'pred', c, 'count', int(cm2[r,c]))
                printed += 1
                if printed >= 10:
                    break
    print()
PY

## ✅ Full GridSearchCV retraining on cleaned datasets completed

The cleaned datasets were used:
- X_classification_no_leak.npy
- feature_names_no_leak.json
- X_localization_no_leak.npy
- localization_feature_names_no_leak.json

The retrained metrics are saved in:
- leak_detection_metrics_no_leak.json
- localization_metrics_no_leak.json

---

## Leak detection (cleaned, GridSearchCV)

- `selected_model`: `RandomForest`
- `best_params`: `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 100}`
- `test accuracy`: `1.0000`
- `macro avg`: precision `1.0000`, recall `1.0000`, f1 `1.0000`
- `weighted avg`: precision `1.0000`, recall `1.0000`, f1 `1.0000`
- `confusion matrix`: perfect `4 x 4` diagonal, `540/540` correct

Top 10 feature importances:
1. `total_abs_Q1` — `0.231038`
2. `total_abs_Q2` — `0.214455`
3. `std_Hm` — `0.141578`
4. `mean_abs_delta_Q` — `0.125906`
5. `mean_Hm` — `0.115840`
6. `sensor_24_Q1` — `0.004104`
7. `sensor_34_Q2` — `0.003914`
8. `sensor_34_Q1` — `0.003145`
9. `sensor_6_Q1` — `0.003105`
10. `sensor_7_delta_Q` — `0.003038`

**Analysis**
- After removing leak-derived features, the detector still achieves perfect accuracy on this cleaned test split.
- The model now relies on hydraulic summary features and sensor flow/pressure values, not direct `Q_leak` signals.
- This is strong evidence that the dataset still contains very separable leak/no-leak signal from the remaining engineered features.

---

## Localization (cleaned, GridSearchCV)

- `selected_model`: `RandomForest`
- `best_params`: `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'n_estimators': 200}`
- `test accuracy`: `0.924074074074074`
- `macro avg`: precision `0.9152`, recall `0.9013`, f1 `0.9027`
- `weighted avg`: precision `0.9304`, recall `0.9241`, f1 `0.9228`
- `confusion matrix`: `35 x 35`, `499/540` correct

Largest misclassifications:
- true `11` → pred `10`: `5`
- true `16` → pred `31`: `3`
- true `15` → pred `24`: `2`
- true `5` → pred `4`: `2`
- true `24` → pred `8`: `2`

Top 10 feature importances:
1. `Q2_mean` — `0.110175`
2. `imbalance_Q_total` — `0.096715`
3. `Q1_mean` — `0.096671`
4. `residual_23` — `0.022374`
5. `residual_26` — `0.022292`
6. `residual_10` — `0.022119`
7. `residual_6` — `0.022085`
8. `residual_32` — `0.021710`
9. `residual_12` — `0.021191`
10. `Hm_std` — `0.021159`

**Analysis**
- Localization remains realistic and strong after removing leak-derived features.
- Top features are physical observables: mean flows, imbalance, residuals, and pressure variation.
- This is the right behavior for a deployed localization model.

---

## Bottom line

- Yes — full GridSearchCV retraining on cleaned datasets ran successfully.
- Leak detection still shows near-perfect test performance, now based on cleaned features.
- Localization stays strong at ~92.4% test accuracy with no explicit leak-flow leakage.

If you want, I can next:
1. run an ablation on the remaining global summary features, or  
2. export the full per-class classification report and confusion matrix for both models to CSV.