from pydantic import BaseModel, field_validator


class FreelancerEmbedIP(BaseModel):
    bio      : str
    skills   : list[str]
    job_title: str

    @field_validator("skills")
    @classmethod
    def skills_not_empty(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("skills list must contain at least one non-empty string.")
        return cleaned

    @field_validator("bio", "job_title")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()


class JobEmbedIP(BaseModel):
    description: str
    skills     : list[str]
    job_title  : str

    @field_validator("skills")
    @classmethod
    def skills_not_empty(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("skills list must contain at least one non-empty string.")
        return cleaned

    @field_validator("description", "job_title")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()


class EmbeddingOP(BaseModel):
    success      : bool
    message      : str
    enriched_text: str
    embedding    : list[float]
    embedding_dim: int