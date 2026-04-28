# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pure-Python compressed suffix tree for suffix decoding speculation.

This implements the core data structure used by suffix decoding: an online
compressed suffix tree that supports incremental token insertion, context
matching against a recent token tail, and greedy path speculation to propose
candidate continuations ranked by frequency.

The algorithm is based on the suffix tree described in Arctic Inference
(https://github.com/snowflakedb/ArcticInference).
"""

import heapq
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SuffixDecodingDraft:
    """Output of suffix tree speculation."""

    token_ids: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @property
    def is_tree(self) -> bool:
        """Whether this draft contains branching structure."""
        return any(parent != idx - 1 for idx, parent in enumerate(self.parents))


class SuffixTreeNode:
    """A node in the compressed suffix tree.

    Each node represents a compressed span of tokens. The actual token values
    are stored in the tree's sequence storage and referenced via `ref_seq` and
    `ref_idx`. The first token of the span is cached in `token` for child
    lookup purposes.
    """

    __slots__ = ("token", "count", "length", "ref_seq", "ref_idx", "children", "parent", "endpoints")

    def __init__(
        self,
        token: int = 0,
        count: int = 0,
        length: int = 0,
        ref_seq: int = 0,
        ref_idx: int = -1,
    ):
        self.token = token
        self.count = count
        self.length = length
        self.ref_seq = ref_seq
        self.ref_idx = ref_idx
        self.children: dict[int, SuffixTreeNode] = {}
        self.parent: SuffixTreeNode | None = None
        self.endpoints: dict[int, int] = {}

    @property
    def head_child(self) -> "SuffixTreeNode | None":
        """Return the child with the highest visit count (greedy pick)."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.count)


class SuffixTree:
    """Compressed suffix tree with online insertion, context matching, and
    greedy path speculation.

    The tree maintains all suffixes (up to `max_depth` length) of every
    appended token sequence. Path compression stores token spans by reference
    into the original sequences, keeping memory close to linear in total
    sequence length.

    Args:
        max_depth: Maximum suffix length tracked in the tree. Longer suffixes
            are dropped from the sliding window of active insertion points.
    """

    def __init__(self, max_depth: int = 64):
        self._max_depth = max_depth
        self._root = SuffixTreeNode(count=0, length=0)
        self._seqs: dict[int, list[int]] = {}
        self._active_nodes: dict[int, deque] = {}

    def extend(self, seq_id: int, tokens: list[int]) -> None:
        """Append multiple tokens to a sequence, updating the tree incrementally."""
        for token in tokens:
            self.append(seq_id, token)

    def append(self, seq_id: int, token: int) -> None:
        """Append a single token to a sequence, updating all active suffixes.

        This is the core online insertion algorithm. For each new token, it:
        1. Pushes the root as a new active suffix start.
        2. Trims the active suffix window to `max_depth`.
        3. Extends every active suffix by the new token, handling three cases:
           - No matching child: extend leaf or create new child.
           - Child exists with count == node.count - 1: fuse/extend.
           - Child exists with multiple paths: move into child, possibly split.
        """
        if seq_id not in self._seqs:
            self._seqs[seq_id] = []
            self._active_nodes[seq_id] = deque()

        seq = self._seqs[seq_id]
        active = self._active_nodes[seq_id]

        # Push root as new active suffix start
        active.append(self._root)
        self._root.endpoints[seq_id] = len(seq)
        self._root.count += 1

        # Maintain max_depth active nodes
        if len(active) > self._max_depth:
            active.popleft()

        seq.append(token)
        seq_len = len(seq)

        for i in range(len(active)):
            node = active[i]
            child = node.children.get(token)

            if child is None:
                # CASE 1: No existing child for this token
                if node.count == 1 and node is not self._root:
                    # Case 1a: Leaf node with count=1 — extend in place
                    node.length += 1
                    node.endpoints[seq_id] = seq_len
                else:
                    # Case 1b: Create new child node
                    new_child = SuffixTreeNode(
                        token=token,
                        count=1,
                        length=1,
                        ref_seq=seq_id,
                        ref_idx=seq_len - 1,
                    )
                    new_child.parent = node
                    new_child.endpoints[seq_id] = seq_len
                    node.children[token] = new_child
                    node.endpoints.pop(seq_id, None)
                    active[i] = new_child

            elif node.count == child.count + 1 and node is not self._root:
                # CASE 2: Child exists, node.count == child.count + 1
                # This means all other suffixes go through the child;
                # only the current suffix ends at node.
                if child.length == 1:
                    # Case 2a: Child length is 1 — fuse child into node
                    parent = node.parent
                    child.count += 1
                    child.token = node.token
                    child.length = node.length + 1
                    child.ref_seq = seq_id
                    child.ref_idx = seq_len - child.length
                    child.endpoints[seq_id] = seq_len
                    child.parent = parent

                    # Replace node with child in parent's children map
                    if parent is not None:
                        parent.children[child.token] = child

                    active[i] = child
                else:
                    # Case 2b: Child length > 1 — extend node, shrink child
                    node.length += 1
                    node.endpoints[seq_id] = seq_len
                    node.ref_seq = seq_id
                    node.ref_idx = seq_len - node.length

                    child.length -= 1
                    child.ref_idx += 1
                    old_token = child.token
                    child.token = self._seqs[child.ref_seq][child.ref_idx]

                    if child.token != old_token:
                        # Re-key in parent's children dict
                        del node.children[old_token]
                        node.children[child.token] = child
            else:
                # CASE 3: Child exists, move active node into child
                if child.length == 1:
                    # Case 3a: Child length is 1 — just move into it
                    node.endpoints.pop(seq_id, None)
                    child.endpoints[seq_id] = seq_len
                    child.count += 1
                    active[i] = child
                else:
                    # Case 3b: Child length > 1 — split child
                    new_node = SuffixTreeNode(
                        token=token,
                        count=child.count + 1,
                        length=1,
                        ref_seq=seq_id,
                        ref_idx=seq_len - 1,
                    )
                    new_node.parent = node
                    node.children[token] = new_node

                    # Shrink original child and re-parent under new_node
                    child.length -= 1
                    child.ref_idx += 1
                    child.token = self._seqs[child.ref_seq][child.ref_idx]
                    child.parent = new_node
                    new_node.children[child.token] = child

                    node.endpoints.pop(seq_id, None)
                    new_node.endpoints[seq_id] = seq_len

                    active[i] = new_node

    def match_context(self, context: list[int]) -> tuple[SuffixTreeNode, int] | None:
        """Match a token sequence against the tree.

        Walks the tree following `context` tokens through compressed node
        spans. Returns `(node, idx)` where `idx` is the position within
        `node`'s compressed span where the match ends, or `None` if no match.
        """
        node = self._root
        idx = 0
        ref_data: list[int] | None = None

        for token in context:
            if idx >= node.length:
                child = node.children.get(token)
                if child is None:
                    return None
                node = child
                ref_data = self._seqs[node.ref_seq]
                idx = 0

            if ref_data is None:
                return None
            if ref_data[node.ref_idx + idx] != token:
                return None
            idx += 1

        return (node, idx)

    def speculate_path(
        self,
        node: SuffixTreeNode,
        idx: int,
        max_spec_tokens: int,
        min_token_prob: float,
    ) -> SuffixDecodingDraft:
        """Generate a linear-chain draft following the highest-count path.

        Starting from position `idx` within `node`, continues along the
        compressed span and then greedily follows the child with the highest
        visit count. Token probability is estimated as
        ``child.count / parent.count``.
        """
        draft = SuffixDecodingDraft()
        prob = 1.0
        ref_data = self._seqs[node.ref_seq]

        while len(draft.token_ids) < max_spec_tokens and prob >= min_token_prob:
            if idx < node.length:
                parent_idx = len(draft.token_ids) - 1
                draft.parents.append(parent_idx)
                draft.token_ids.append(ref_data[node.ref_idx + idx])
                draft.probs.append(prob)
                draft.depths.append(len(draft.token_ids))
                draft.score += prob
                idx += 1
            else:
                child = node.head_child
                if child is None:
                    break
                prob *= child.count / node.count
                node = child
                ref_data = self._seqs[node.ref_seq]
                idx = 0

        return draft

    def speculate_tree_path(
        self,
        node: SuffixTreeNode,
        idx: int,
        max_spec_tokens: int,
        min_token_prob: float,
        branch_factor: int,
    ) -> SuffixDecodingDraft:
        """Generate a branching draft from a matched suffix tree position.

        The output is a topologically sorted tree: every token's parent appears
        before the token itself, and parent ``-1`` means the token is a direct
        child of the current generation context. Expansion is probability-ranked
        and bounded by ``branch_factor`` children per suffix-tree node.
        """
        if max_spec_tokens <= 0:
            return SuffixDecodingDraft()

        draft = SuffixDecodingDraft()
        branch_factor = max(branch_factor, 1)
        counter = 0
        queue = [(-1.0, counter, node, idx, -1, 0)]

        while queue and len(draft.token_ids) < max_spec_tokens:
            neg_prob, _, node, idx, parent_idx, depth = heapq.heappop(queue)
            prob = -neg_prob
            if prob < min_token_prob:
                continue

            if idx < node.length:
                ref_data = self._seqs[node.ref_seq]
                draft_idx = len(draft.token_ids)
                draft.parents.append(parent_idx)
                draft.token_ids.append(ref_data[node.ref_idx + idx])
                draft.probs.append(prob)
                draft.depths.append(depth + 1)
                draft.score += prob
                counter += 1
                heapq.heappush(queue, (-prob, counter, node, idx + 1, draft_idx, depth + 1))
                continue

            if node.count == 0:
                continue

            children = sorted(node.children.values(), key=lambda child: child.count, reverse=True)[:branch_factor]
            for child in children:
                child_prob = prob * child.count / node.count
                if child_prob < min_token_prob:
                    continue
                counter += 1
                heapq.heappush(queue, (-child_prob, counter, child, 0, parent_idx, depth))

        return draft

    def speculate(
        self,
        context: list[int],
        max_spec_tokens: int,
        min_token_prob: float = 0.1,
        max_spec_factor: float = 3.0,
        max_spec_offset: float = 5.0,
    ) -> SuffixDecodingDraft:
        """Produce a speculative draft by matching context suffixes.

        Tries progressively shorter suffixes of `context` against the tree.
        For each match, generates a greedy path draft and keeps the
        highest-scoring one. Longer matches are allowed to produce longer
        drafts (controlled by `max_spec_factor` and `max_spec_offset`).

        Args:
            context: Recent token tail to match against the tree.
            max_spec_tokens: Hard cap on number of speculated tokens.
            min_token_prob: Stop speculation when token probability drops below
                this threshold.
            max_spec_factor: Soft cap factor — draft length <=
                ``match_len * max_spec_factor + max_spec_offset``.
            max_spec_offset: Soft cap offset added to the factor-based limit.

        Returns:
            The best-scoring `SuffixDecodingDraft`. May have empty
            `token_ids` if no suffix matched.
        """
        best = SuffixDecodingDraft()

        for match_len in range(1, len(context)):
            sub_context = context[-match_len:]
            result = self.match_context(sub_context)
            if result is None:
                break

            node, idx = result
            cap = int(match_len * max_spec_factor + max_spec_offset + 1e-6)
            effective_max = min(max_spec_tokens, max(cap, 0))

            draft = self.speculate_path(node, idx, effective_max, min_token_prob)
            if draft.score >= best.score:
                draft.match_len = match_len
                best = draft

        return best

    def speculate_tree(
        self,
        context: list[int],
        max_spec_tokens: int,
        min_token_prob: float = 0.1,
        branch_factor: int = 2,
        max_spec_factor: float = 3.0,
        max_spec_offset: float = 5.0,
    ) -> SuffixDecodingDraft:
        """Produce a branching speculative draft by matching context suffixes.

        This follows the same suffix matching policy as :meth:`speculate`, but
        expands multiple high-frequency continuations from each matched tree
        node. The result can be verified with a tree attention mask.
        """
        best = SuffixDecodingDraft()

        for match_len in range(1, len(context)):
            sub_context = context[-match_len:]
            result = self.match_context(sub_context)
            if result is None:
                break

            node, idx = result
            cap = int(match_len * max_spec_factor + max_spec_offset + 1e-6)
            effective_max = min(max_spec_tokens, max(cap, 0))

            draft = self.speculate_tree_path(node, idx, effective_max, min_token_prob, branch_factor)
            if draft.score >= best.score:
                draft.match_len = match_len
                best = draft

        return best


class SuffixDecodingCache:
    """Persistent cache for suffix decoding across multiple generation calls.

    Maintains a pool of per-request suffix trees from completed requests,
    allowing future requests to draft from previously seen token sequences.
    Each active request also gets a local suffix tree seeded with its prompt
    tokens. Speculation queries the local tree and all cached global trees,
    returning the highest-scoring draft.

    When the cache reaches capacity, the oldest request's tree is evicted
    entirely, ensuring clean memory reclamation without requiring node-level
    deletion from a shared tree.

    Usage::

        cache = SuffixDecodingCache(max_depth=64, max_cached_requests=100)

        for prompt in prompts:
            ids = tokenizer.encode(prompt, return_tensors="pt")
            output = model.generate(
                ids,
                suffix_decoding_num_tokens=10,
                suffix_decoding_cache=cache,
            )

    Args:
        max_depth: Maximum suffix tree depth for both local and global trees.
        max_cached_requests: Maximum number of completed responses to keep in
            the global cache. When exceeded, oldest responses are evicted.
            Set to 0 to disable the global cache entirely (local-only mode).
            Set to -1 for unlimited caching.
        source_mode: Which candidate source(s) to use when speculating.
            ``"local_and_global"`` compares the active request's local tree
            against all cached global trees and keeps the highest-scoring
            draft. ``"local_only"`` uses only the active request's tree.
            ``"global_only"`` uses only cached trees from prior requests.
    """

    def __init__(
        self,
        max_depth: int = 64,
        max_cached_requests: int = -1,
        source_mode: str = "local_and_global",
    ):
        if source_mode not in {"local_and_global", "local_only", "global_only"}:
            raise ValueError(
                "source_mode must be one of: "
                "'local_and_global', 'local_only', 'global_only'"
            )
        self._max_depth = max_depth
        self._max_cached_requests = max_cached_requests
        self._source_mode = source_mode
        self._global_trees: dict[int, SuffixTree] = {}
        self._local_trees: dict[int, SuffixTree] = {}
        self._next_seq_id = 1
        self._cached_order: list[int] = []

    def start_request(self, prompt_token_ids: list[int]) -> int:
        """Start a new request, creating a local tree seeded with prompt tokens.

        Args:
            prompt_token_ids: The prompt tokens to seed the local tree with.

        Returns:
            A request ID to pass to subsequent calls.
        """
        req_id = self._next_seq_id
        self._next_seq_id += 1

        local_tree = SuffixTree(max_depth=self._max_depth)
        local_tree.extend(0, prompt_token_ids)
        self._local_trees[req_id] = local_tree

        # Allocate a global tree slot if caching is enabled
        if self._max_cached_requests != 0:
            self._evict_if_needed()
            self._global_trees[req_id] = SuffixTree(max_depth=self._max_depth)
            self._cached_order.append(req_id)

        return req_id

    def add_active_response(self, req_id: int, token_ids: list[int]) -> None:
        """Append accepted output tokens to the local and global trees.

        Only call this with verified (accepted) tokens, never speculative ones.

        Args:
            req_id: The request ID returned by :meth:`start_request`.
            token_ids: Accepted output token IDs to append.
        """
        if req_id in self._local_trees:
            self._local_trees[req_id].extend(0, token_ids)

        if req_id in self._global_trees:
            self._global_trees[req_id].extend(0, token_ids)

    def speculate(
        self,
        req_id: int,
        context: list[int],
        max_spec_tokens: int,
        min_token_prob: float = 0.1,
        max_spec_factor: float = 3.0,
        max_spec_offset: float = 5.0,
    ) -> SuffixDecodingDraft:
        """Run speculation against local and all cached global trees, return the best.

        Args:
            req_id: The request ID returned by :meth:`start_request`.
            context: Recent token tail to match.
            max_spec_tokens: Hard cap on speculated tokens.
            min_token_prob: Minimum probability threshold.
            max_spec_factor: Soft cap factor.
            max_spec_offset: Soft cap offset.

        Returns:
            The highest-scoring draft from any tree.
        """
        spec_kwargs = {
            "context": context,
            "max_spec_tokens": max_spec_tokens,
            "min_token_prob": min_token_prob,
            "max_spec_factor": max_spec_factor,
            "max_spec_offset": max_spec_offset,
        }

        best = SuffixDecodingDraft()

        if self._source_mode in {"local_and_global", "local_only"} and req_id in self._local_trees:
            local_draft = self._local_trees[req_id].speculate(**spec_kwargs)
            if local_draft.score >= best.score:
                best = local_draft

        if self._source_mode in {"local_and_global", "global_only"}:
            for cached_id, tree in self._global_trees.items():
                if cached_id == req_id:
                    continue  # skip own global tree, local tree covers it
                draft = tree.speculate(**spec_kwargs)
                if draft.score >= best.score:
                    best = draft

        return best

    def speculate_tree(
        self,
        req_id: int,
        context: list[int],
        max_spec_tokens: int,
        min_token_prob: float = 0.1,
        branch_factor: int = 2,
        max_spec_factor: float = 3.0,
        max_spec_offset: float = 5.0,
    ) -> SuffixDecodingDraft:
        """Run branching speculation against local/global suffix trees."""
        spec_kwargs = {
            "context": context,
            "max_spec_tokens": max_spec_tokens,
            "min_token_prob": min_token_prob,
            "branch_factor": branch_factor,
            "max_spec_factor": max_spec_factor,
            "max_spec_offset": max_spec_offset,
        }

        best = SuffixDecodingDraft()

        if self._source_mode in {"local_and_global", "local_only"} and req_id in self._local_trees:
            local_draft = self._local_trees[req_id].speculate_tree(**spec_kwargs)
            if local_draft.score >= best.score:
                best = local_draft

        if self._source_mode in {"local_and_global", "global_only"}:
            for cached_id, tree in self._global_trees.items():
                if cached_id == req_id:
                    continue
                draft = tree.speculate_tree(**spec_kwargs)
                if draft.score >= best.score:
                    best = draft

        return best

    def stop_request(self, req_id: int) -> None:
        """End a request, removing its local tree but keeping global cache.

        Args:
            req_id: The request ID returned by :meth:`start_request`.
        """
        self._local_trees.pop(req_id, None)

    def _evict_if_needed(self) -> None:
        """Evict the oldest cached response tree if at capacity."""
        if self._max_cached_requests < 0:
            return
        while len(self._cached_order) >= self._max_cached_requests:
            if not self._cached_order:
                break
            evicted_id = self._cached_order.pop(0)
            self._global_trees.pop(evicted_id, None)
