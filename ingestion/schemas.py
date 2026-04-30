from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime, timezone


class LineItem(BaseModel):
    """Represents a single line item or service in a financial document."""

    description: str = Field(description="Description of the item or service")
    quantity: float = Field(default=1.0, description="Number of units")
    unit_price: float = Field(description="Price per single unit")
    total_price: float = Field(description="Total price for this line (qty * unit_price)")
    tax_rate: Optional[float] = Field(
        default=14.0,
        description="VAT percentage — default is 14% for Egypt"
    )

    @field_validator("total_price")
    @classmethod
    def total_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("total_price cannot be negative")
        return round(v, 2)


class ExtractedDocument(BaseModel):
    """Primary schema for all processed financial documents in the system."""

    # FIX 1: added "unknown" to Literal so classifier output never crashes schema
    doc_type: Literal["invoice", "balance_sheet", "tax_return", "contract", "unknown"] = Field(
        description="Document category identified by the classifier"
    )

    # FIX 2: added default=None to all Optional fields — they were effectively required before
    vendor_name: Optional[str] = Field(default=None, description="Legal name of the issuing entity")
    tax_id: Optional[str] = Field(default=None, description="Egyptian Tax Registration Number (9 digits)")
    currency: str = Field(default="EGP", description="Currency code (EGP, USD, EUR)")
    date: Optional[str] = Field(default=None, description="Issue date found on the document")

    line_items: List[LineItem] = Field(
        default_factory=list,
        description="All products or services listed in the document"
    )

    subtotal: float = Field(default=0.0, description="Total amount before taxes")
    total_tax: float = Field(default=0.0, description="Sum of all applied taxes")
    total_amount: float = Field(default=0.0, description="Grand total including taxes")

    raw_text: str = Field(description="Full unprocessed text from the OCR pipeline")
    cleaned_raw_text: Optional[str] = Field(
        default=None,
        description="Post-processed OCR text with corrected common OCR noise"
    )

    # FIX 3: confidence is per-field now, not just one number for the whole doc
    extraction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the extraction process (0.0 to 1.0)"
    )
    validation_status: bool = Field(
        default=False,
        description="True if math checks pass (subtotal + tax == total)"
    )

    # NEW: track when this document was processed — useful for audit logging later
    processed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of when this document was processed"
    )

    # NEW: flag for Phase 3 HITL escalation
    requires_human_review: bool = Field(
        default=False,
        description="True if confidence is too low for automated auditing"
    )

    # Document category for agent routing
    doc_category: str = Field(
        default="invoice",
        description="'invoice', 'financial_statement', or 'unknown' — routes audit rules"
    )
