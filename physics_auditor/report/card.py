from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReportRow:
    model: str
    law: str
    probe: str
    metric: str
    value: float
    n_pairs: int


@dataclass
class ReportCard:
    _rows: list[ReportRow] = field(default_factory=list)

    def add_row(
        self,
        model: str,
        law: str,
        probe: str,
        metric: str,
        value: float,
        n_pairs: int,
    ) -> None:
        self._rows.append(
            ReportRow(
                model=model,
                law=law,
                probe=probe,
                metric=metric,
                value=value,
                n_pairs=n_pairs,
            )
        )

    @property
    def rows(self) -> list[ReportRow]:
        return list(self._rows)

    def to_markdown(self) -> str:
        header = "| model | law | probe | metric | value | n_pairs |"
        separator = "| --- | --- | --- | --- | --- | --- |"
        lines = [header, separator]
        for row in self._rows:
            lines.append(
                f"| {row.model} | {row.law} | {row.probe} | {row.metric} | "
                f"{row.value:.3f} | {row.n_pairs} |"
            )
        return "\n".join(lines)
