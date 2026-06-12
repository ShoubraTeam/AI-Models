# -----------------------------------------------------
# Configurations related to data/inputs/outputs
# -----------------------------------------------------

# all features
FEATURE_IDENITY_RECOGNITION         = "identity_recognition"
FEATURE_JOB_RECOMMENDATION_SYSTEM   = "job_recommendation_system"
FEATURE_PROFILE_ANALYSIS            = "profile_analysis"
FEATURE_JOB_DESCRIPTION_ENHANCEMENT = "job_description_enhancement"
FEATURE_PROPOSAL_REJECTION_REASONS  = "proposal_rejection_reasons"


# job desc
FEATURE_ALLOWED_FEATURES = [
    FEATURE_IDENITY_RECOGNITION,
    FEATURE_JOB_RECOMMENDATION_SYSTEM,
    FEATURE_PROFILE_ANALYSIS,
    FEATURE_JOB_DESCRIPTION_ENHANCEMENT,
    FEATURE_PROPOSAL_REJECTION_REASONS,
]


JOB_DESC_TOOLS_DETECTION              = "job_desc_tools_detection"
JOB_DESC_TOOLS_RECOMMENDATION         = "job_desc_tools_recommendation"
JOB_DESC_JOB_DESCRIPTION_ENHANCEMENT  = "job_desc_job_description_enhancement"
JOB_DESCRIPTION_ALLOWED_TASKS = [
    JOB_DESC_TOOLS_DETECTION,
    JOB_DESC_TOOLS_RECOMMENDATION,
    JOB_DESC_JOB_DESCRIPTION_ENHANCEMENT
]


# prr
PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION = "PRR_job_features_extraction"
PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS       = "PRR_proposal_analysis"

PROPOSAL_REJECTION_REASONS_ALLOWED_TASKS = [
    PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION,
    PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS
]


# profile scorer
PROFILE_SCORER_FEATURES_EXTRACTION  = "profile_scorer_features_extraction"
PROFILE_SCORER_FINAL_ANALYSIS       = "profile_scorer_final_analysis"

PROFILE_SCORER_ALLOWED_TASKS = [
    PROFILE_SCORER_FEATURES_EXTRACTION,
    PROFILE_SCORER_FINAL_ANALYSIS
]


# recommendation system
RS_FREELANCER_EMBEDDING = "RS_freelancer_embedding"
RS_JOB_EMBEDDING        = "RS_job_embedding"

RS_ALLOWED_TASKS = [
    RS_FREELANCER_EMBEDDING,
    RS_JOB_EMBEDDING,
]