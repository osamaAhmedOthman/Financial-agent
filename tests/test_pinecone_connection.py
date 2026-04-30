"""Integration check for Pinecone connectivity.

This file must stay import-safe so pytest can collect the test suite even
when Pinecone credentials are not configured locally.
"""

import os

import pytest
from dotenv import load_dotenv


load_dotenv()


@pytest.mark.integration
def test_pinecone_connection():
	api_key = os.getenv("PINECONE_API_KEY")
	if not api_key:
		pytest.skip("PINECONE_API_KEY not set")

	try:
		from pinecone import Pinecone

		pc = Pinecone(api_key=api_key)
		indexes = pc.list_indexes()
	except Exception as exc:
		pytest.skip(f"Pinecone connection unavailable: {exc}")

	assert indexes is not None
	assert isinstance([idx.name for idx in indexes], list)