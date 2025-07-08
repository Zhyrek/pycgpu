# GPU DOF Indexing Fix

## Root Cause Identified

The GPU code has a critical indexing mismatch between what the compiled functions expect and what's in the DOF array.

### The Problem:

1. **Model expectations**: The BCC_A2 model only uses temperature (T), so its compiled functions expect:
   - `x[0] = T`
   - `x[1] = Y(BCC_A2,0,NB)`
   - `x[2] = Y(BCC_A2,0,TI)`

2. **DOF array structure**: The phase record stores ALL state variables:
   - `dof[0] = N`
   - `dof[1] = P`
   - `dof[2] = T`
   - `dof[3] = Y(BCC_A2,0,NB)`
   - `dof[4] = Y(BCC_A2,0,TI)`

3. **Current GPU behavior**: Passes the full DOF array directly, causing:
   - Function reads `x[0]` expecting T=1000, gets N=1.0
   - Function reads `x[1]` expecting Y(NB)=0.6, gets P=101325
   - Function reads `x[2]` expecting Y(TI)=0.4, gets T=1000

### The Solution:

The GPU code needs to:
1. Extract only the state variables that the model actually uses
2. Concatenate them with the site fractions
3. Pass this reordered array to the compiled functions

### Implementation:

Create a temporary array that maps from the phase record's DOF layout to what the model expects:
- For each model state variable, find its index in the phase record's state variables
- Copy those values to the temporary array
- Then copy all site fractions

This matches what the CPU must be doing internally when it builds the compiled functions with only the model's state variables.