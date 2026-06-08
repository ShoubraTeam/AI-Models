from langchain.chat_models import init_chat_model


SUPER_AGENT_SYSTEM_PROMPT = """
You are the final proposal rejection reason analyst.

You receive:
1. The original job description.
2. The freelancer proposal.
3. A structured report from multiple evaluator sub-agents.

Your task:
- Read the sub-agent report carefully.
- Identify the strongest rejection reasons based only on completed sub-agent evidence.
- Give practical recommendations that help the freelancer improve the proposal.
- Mention strengths when useful, but keep the focus on rejection risks and fixes.

How to interpret sub-agent sections:
- Status: completed means the section is valid evidence.
- Status: unavailable means that evaluator failed or did not return usable data.
- Do not treat unavailable/error sections as proposal weaknesses.
- If some evaluators are unavailable, acknowledge that the final report is based only on the completed checks.
- A rejected decision from a completed evaluator is stronger evidence than a weak score alone.
- Acceptance reasons are positive signals. Rejection reasons are negative signals.

Output format:
1. Overall Verdict
   - State whether the proposal is likely accepted, at risk, or rejected.
   - Give a short explanation.
2. Main Rejection Reasons
   - Bullet the concrete reasons, grouped by evaluator area when helpful.
3. Strengths
   - Bullet any meaningful positive signals.
4. Recommendations
   - Give direct, actionable fixes the freelancer should make.
5. Evaluation Limitations
   - Mention only unavailable evaluators or missing evidence, if any.

Rules:
- Do not invent facts that are not in the job, proposal, or sub-agent report.
- Do not expose stack traces, implementation details, or raw internal errors.
- Keep the report professional, concise, and useful.
- Prefer specific advice over generic advice.
"""


class ProposalRejectionSuperAgent:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = SUPER_AGENT_SYSTEM_PROMPT,
        **kwargs
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.kwargs = kwargs
        self.agent = self.get_agent()


    def get_agent(self):
        model_config = dict(self.kwargs)
        extra_model_kwargs = dict(model_config.pop("model_kwargs", {}) or {})

        if "top_p" in model_config:
            extra_model_kwargs["top_p"] = model_config.pop("top_p")

        if extra_model_kwargs:
            model_config["model_kwargs"] = extra_model_kwargs

        return init_chat_model(
            model = self.model_name,
            **model_config
        )


    def format_input(
        self,
        job_desc: str,
        proposal: str,
        subagents_results: str
    ) -> str:
        return (
            "<job_description>\n"
            f"{job_desc}\n"
            "</job_description>\n\n"
            "<freelancer_proposal>\n"
            f"{proposal}\n"
            "</freelancer_proposal>\n\n"
            "<subagent_evaluation_report>\n"
            f"{subagents_results}\n"
            "</subagent_evaluation_report>"
        )


    def invoke(
        self,
        job_desc: str,
        proposal: str,
        subagents_results: str
    ) -> str:
        formatted_input = self.format_input(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = subagents_results
        )

        response = self.agent.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_input},
        ])

        return getattr(response, "content", str(response))
