from __future__ import annotations

from server_utils.ltx_video_normalization import plan_temporal_chunks


def test_plan_temporal_chunks_returns_single_chunk_when_not_needed() -> None:
    chunks = plan_temporal_chunks(total_frames=41, max_chunk_frames=41, overlap_frames=17)

    assert len(chunks) == 1
    assert chunks[0].start_frame == 0
    assert chunks[0].frame_count == 41
    assert chunks[0].keep_start_frame == 0
    assert chunks[0].keep_frame_count == 41


def test_plan_temporal_chunks_covers_all_frames_without_gaps_or_duplicates() -> None:
    total_frames = 233
    chunks = plan_temporal_chunks(total_frames=total_frames, max_chunk_frames=113, overlap_frames=17)

    assert len(chunks) >= 2

    covered: list[int] = []
    for chunk in chunks:
        covered.extend(
            range(
                chunk.start_frame + chunk.keep_start_frame,
                chunk.start_frame + chunk.keep_start_frame + chunk.keep_frame_count,
            )
        )

    assert covered == list(range(total_frames))


def test_plan_temporal_chunks_respects_chunk_budget() -> None:
    chunks = plan_temporal_chunks(total_frames=233, max_chunk_frames=113, overlap_frames=17)

    assert all(chunk.frame_count <= 113 for chunk in chunks)
    assert all(chunk.keep_frame_count > 0 for chunk in chunks)
