from dataclasses import dataclass

from typing import Tuple


@dataclass
class Fact:
	"""Temporal fact class."""
	head: str
	rel: str
	tail: str
	time: str

	def quadruple(
			self, 
			_format: str = "normal",
	) -> Tuple[str, str, str, str]:
		"""Quadruple representation.
		
		Args:
			_format (str): select from normal/inverse/swap.
		
		Returns:
			Quadruple: (head, rel, tail, time)
		"""
		head = self.head
		rel = self.rel
		tail = self.tail
		time = self.time
		if _format != "normal":
			head, tail = tail, head
		if _format == "inverse":
			rel = f"inverse {rel}"
		return head, rel, tail, time
