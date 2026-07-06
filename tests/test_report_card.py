from physics_auditor.report.card import ReportCard, ReportRow


def test_add_and_retrieve_row():
    card = ReportCard()
    card.add_row("oracle", "support", "behavioural", "auroc", 1.0, 5)
    assert len(card.rows) == 1
    row = card.rows[0]
    assert row == ReportRow(
        model="oracle",
        law="support",
        probe="behavioural",
        metric="auroc",
        value=1.0,
        n_pairs=5,
    )


def test_markdown_has_header_and_row():
    card = ReportCard()
    card.add_row("oracle", "support", "behavioural", "auroc", 1.0, 5)
    md = card.to_markdown()
    assert "| model | law | probe | metric | value | n_pairs |" in md
    assert "oracle" in md
    assert "support" in md
    assert "1.000" in md


def test_multiple_rows_in_order():
    card = ReportCard()
    card.add_row("oracle", "support", "behavioural", "auroc", 1.0, 5)
    card.add_row("random", "support", "behavioural", "auroc", 0.5, 5)
    md = card.to_markdown()
    assert md.index("oracle") < md.index("random")
    assert len(card.rows) == 2
    assert card.rows[0].model == "oracle"
    assert card.rows[1].model == "random"
