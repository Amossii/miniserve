# MiniServe Scheduler Design

## 1. Goal

The Scheduler decides which requests are admitted into the active execution set before each inference iteration.

The Scheduler operates on Request metadata.

It does not perform model forward execution or construct GPU tensors.

---

## 2. Request Collections

The Scheduler maintains three logical collections.

### Waiting

Requests that have arrived but have not yet been admitted.

Invariant:

```text
request.status == WAITING
```

Waiting requests use FIFO ordering in Scheduler V1.

### Running

Requests that have been admitted and have not yet finished.

Invariant:

```text
request.status == RUNNING
```

The number of running requests must satisfy:

```text
len(running) <= max_num_seqs
```

### Finished

Requests that completed execution and were removed from the running set.

Finished requests may be retained for result collection and metrics.

Invariant:

```text
request.status == FINISHED
```

---

## 3. Scheduler V1 Capacity

Scheduler V1 uses one capacity constraint:

```text
max_num_seqs
```

The number of free active sequence slots is:

```text
free_slots =
max_num_seqs - len(running)
```

Token-budget-based scheduling is not part of Scheduler V1.

It will be added separately later.

---

## 4. Scheduling Iteration

Each scheduling iteration follows this order:

```text
1. Reclaim finished requests from running.

2. Compute available sequence slots.

3. Admit waiting requests in FIFO order.

4. Build the current scheduled request set.

5. Partition scheduled requests by execution phase.

6. Return a scheduling decision to the execution layer.
```

Finished requests must be reclaimed before admission so their active slots can immediately be reused.

---

## 5. Admission

A waiting request may be admitted when:

```text
len(running) < max_num_seqs
```

Admission performs the lifecycle transition:

```text
WAITING
   ↓
RUNNING
```

The Scheduler, rather than the ModelRunner, owns this transition.

Scheduler V1 uses FIFO admission.

No priority, preemption, or fairness policy beyond FIFO is implemented.

---

## 6. Schedule Output

A scheduling decision should conceptually contain:

```text
scheduled_requests
prefill_requests
decode_requests
newly_admitted
newly_finished
```

For Scheduler V1:

```text
scheduled_requests == running_requests
```

because every running request is executed every iteration.

This distinction is retained because future token budgets or preemption may cause only a subset of running requests to be scheduled.

---

## 7. Prefill and Decode

Lifecycle state and execution phase remain separate.

Examples:

```text
WAITING + PREFILL
RUNNING + PREFILL
RUNNING + DECODE
FINISHED
```

After admission, a newly admitted request normally enters:

```text
RUNNING + PREFILL
```

After its prefill execution completes:

```text
RUNNING + DECODE
```

Scheduler V1 may expose separate `prefill_requests` and `decode_requests`, but it does not decide how the two workloads are combined into model forwards.

That responsibility belongs to the execution layer.

---

## 8. Continuous Batch Membership

With:

```text
max_num_seqs = 3
```

an example schedule is:

```text
Iteration 0

waiting:
D E

running:
A B C
```

If B finishes:

```text
Iteration 1 before scheduling

waiting:
D E

running:
A B(finished) C
```

The scheduler first removes B:

```text
running:
A C
```

then admits D:

```text
waiting:
E

running:
A C D
```

Therefore batch membership changes from:

```text
A B C
```

to:

```text
A C D
```

This dynamic reuse of active sequence slots is the basis of continuous batching.

---

## 9. Scheduler V1 Non-Goals

Scheduler V1 does not implement:

* token budgets
* priority scheduling
* request preemption
* recomputation
* starvation prevention policies
* chunked prefill
* memory-aware admission
* KV block allocation
* batching tensor construction
* GPU execution
* distributed scheduling

These features are added only when required by later project stages.

---

## 10. Core Invariants

At all times:

```text
1. A request appears in at most one lifecycle collection.

2. Every waiting request has status WAITING.

3. Every running request has status RUNNING.

4. Finished requests are removed from running before admission.

5. len(running) <= max_num_seqs.

6. Admission order is FIFO.

7. Scheduler does not own model or GPU tensor logic.
```

Any violation should be treated as a scheduler bug.
