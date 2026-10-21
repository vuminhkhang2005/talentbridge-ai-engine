"""
Synthetic & Expert-Annotated Vietnamese CV-JD Dataset Generator
For Specialized Project (Tiểu luận chuyên ngành): AI-Powered CV-JD Matching
Generates 360 diverse, realistic (CV, JD) pairs across 8 IT & Business domains
with ground-truth compatibility scores (0-100) and structured annotations.
"""

import json
import os
import random
from pathlib import Path

random.seed(42)

DOMAINS = [
    {
        "domain": "Backend Java",
        "jd_titles": [
            "Java Spring Boot Backend Engineer",
            "Senior Java Microservices Developer",
            "Fresher Java Web Developer",
        ],
        "core_skills": ["Java", "Spring Boot", "SQL", "PostgreSQL", "RESTful API", "Microservices", "Redis", "Docker", "Git"],
        "nice_skills": ["Kafka", "Kubernetes", "AWS", "CI/CD"],
        "jd_desc": (
            "Phát triển hệ thống Backend hiệu năng cao bằng Java 17/21 và Spring Boot 3. "
            "Thiết kế RESTful API, kiến trúc Microservices, tối ưu hóa truy vấn cơ sở dữ liệu PostgreSQL/MySQL, "
            "xử lý bộ nhớ đệm Redis và hàng đợi tin nhắn Kafka. Triển khai ứng dụng trên Docker và CI/CD."
        ),
    },
    {
        "domain": "Backend .NET",
        "jd_titles": [
            "Fresher .NET Backend Developer",
            "Senior .NET Core Microservices Engineer",
            ".NET Fullstack Software Engineer",
        ],
        "core_skills": ["C#", ".NET Core", "ASP.NET Core", "Entity Framework", "SQL Server", "SQL", "RESTful API", "Git"],
        "nice_skills": ["Docker", "Kubernetes", "Redis", "Azure", "RabbitMQ"],
        "jd_desc": (
            "Tham gia phát triển các dự án phần mềm doanh nghiệp trên nền tảng C#, .NET 8 và ASP.NET Core Web API. "
            "Làm việc với Entity Framework Core, tối ưu hóa SQL Server, thiết kế kiến trúc Clean Architecture và Microservices."
        ),
    },
    {
        "domain": "Frontend React",
        "jd_titles": [
            "Frontend Developer (ReactJS / TypeScript)",
            "Senior Next.js Frontend Engineer",
            "Junior Frontend Web Developer",
        ],
        "core_skills": ["JavaScript", "TypeScript", "React", "Next.js", "Tailwind CSS", "HTML5", "CSS3", "Git"],
        "nice_skills": ["Redux", "GraphQL", "Jest", "Figma", "UI/UX Design"],
        "jd_desc": (
            "Xây dựng giao diện người dùng hiện đại, phản hồi nhanh (Responsive Web) với React 18, Next.js và TypeScript. "
            "Tối ưu hóa Core Web Vitals, kết nối RESTful API/GraphQL và phối hợp chặt chẽ với đội ngũ UI/UX Designer qua Figma."
        ),
    },
    {
        "domain": "AI & Data Engineering",
        "jd_titles": [
            "AI / Machine Learning Engineer (NLP & LLM)",
            "Computer Vision & Deep Learning Engineer",
            "Data Engineer (Python / Spark / SQL)",
        ],
        "core_skills": ["Python", "PyTorch", "Machine Learning", "Deep Learning", "NLP", "SQL", "FastAPI", "Docker"],
        "nice_skills": ["LLM", "RAG", "TensorFlow", "AWS", "Spark", "Airflow"],
        "jd_desc": (
            "Nghiên cứu và huấn luyện các mô hình Học sâu (Deep Learning), Xử lý ngôn ngữ tự nhiên (NLP/LLM) và Thị giác máy tính "
            "bằng Python và PyTorch. Xây dựng pipeline xử lý dữ liệu và triển khai mô hình AI qua FastAPI, Docker trên hạ tầng Cloud."
        ),
    },
    {
        "domain": "Cloud & DevOps",
        "jd_titles": [
            "Senior Cloud Architect (AWS & Kubernetes)",
            "DevOps / Site Reliability Engineer (SRE)",
            "Cloud Infrastructure Engineer",
        ],
        "core_skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Linux", "Terraform", "Git", "Python"],
        "nice_skills": ["Prometheus", "Grafana", "Helm", "Ansible", "Go"],
        "jd_desc": (
            "Thiết kế, vận hành hạ tầng đám mây AWS/GCP tự động hóa bằng Terraform (IaC). Quản trị cụm Kubernetes (EKS), "
            "xây dựng đường ống CI/CD (GitHub Actions/GitLab CI) và hệ thống giám sát Prometheus, Grafana đảm bảo SLA 99.99%."
        ),
    },
    {
        "domain": "Backend Golang",
        "jd_titles": [
            "Backend Golang Engineer (High Concurrency)",
            "Senior Go / Distributed Systems Engineer",
        ],
        "core_skills": ["Go", "Golang", "Microservices", "Redis", "PostgreSQL", "SQL", "Docker", "Kafka", "gRPC"],
        "nice_skills": ["Kubernetes", "AWS", "Prometheus", "CI/CD"],
        "jd_desc": (
            "Phát triển các dịch vụ lõi thanh toán và xử lý giao dịch chịu tải cao bằng ngôn ngữ Go (Golang), gRPC, Kafka và Redis. "
            "Đảm bảo độ trễ thấp (low latency), khả năng mở rộng ngang và an toàn dữ liệu trên PostgreSQL."
        ),
    },
    {
        "domain": "QA & Automation Test",
        "jd_titles": [
            "QA / QC Engineer (Manual & Automation)",
            "Intern / Junior QA Tester",
        ],
        "core_skills": ["Unit Testing", "Automation Test", "Selenium", "Playwright", "API Testing", "Postman", "SQL", "Agile/Scrum"],
        "nice_skills": ["Java", "Python", "CI/CD", "Jira"],
        "jd_desc": (
            "Lập kế hoạch kiểm thử (Test Plan), viết kịch bản kiểm thử (Test Case), thực hiện kiểm thử API bằng Postman "
            "và phát triển khung kiểm thử tự động (Automation Testing) với Playwright/Selenium."
        ),
    },
    {
        "domain": "Business & Digital Marketing",
        "jd_titles": [
            "Chuyên viên Tư vấn & Kinh doanh Giải pháp B2B",
            "Performance Marketing Specialist",
            "Chuyên viên Tuyển dụng & Thu hút Nhân tài (IT Recruiter)",
        ],
        "core_skills": ["Communication", "B2B Sales", "Đàm phán thương mại", "Digital Marketing", "SEO / SEM", "Phân tích dữ liệu", "Teamwork"],
        "nice_skills": ["CRM", "Google Analytics", "Tuyển dụng nhân sự", "HR Business Partner", "English Professional"],
        "jd_desc": (
            "Phát triển khách hàng doanh nghiệp B2B, tư vấn giải pháp công nghệ, đàm phán hợp đồng thương mại và tối ưu hóa "
            "chiến dịch tiếp thị số (Digital Marketing / SEO / SEM) dựa trên phân tích dữ liệu chuyển đổi."
        ),
    },
]

EXP_LEVELS = [
    ("INTERN", 0, "Sinh viên năm cuối hoặc mới tốt nghiệp"),
    ("FRESHER", 0, "Dưới 1 năm kinh nghiệm, nắm vững kiến thức nền tảng"),
    ("JUNIOR", 1, "Từ 1 - 2 năm kinh nghiệm thực tế tại doanh nghiệp"),
    ("MIDDLE", 2, "Từ 2 - 4 năm kinh nghiệm phát triển dự án thực tế"),
    ("SENIOR", 4, "Trên 4 năm kinh nghiệm chuyên sâu và thiết kế kiến trúc hệ thống"),
]


def build_pair(pair_id: int, jd_dom: dict, cv_dom: dict, jd_level: tuple, cv_years: int, match_tier: str) -> dict:
    jd_title = random.choice(jd_dom["jd_titles"])
    req_skills = random.sample(jd_dom["core_skills"], k=min(5, len(jd_dom["core_skills"])))
    opt_skills = random.sample(jd_dom["nice_skills"], k=min(2, len(jd_dom["nice_skills"])))

    if match_tier == "STRONG":
        # Candidate has 80%-100% of required skills + some optional skills
        k_req = max(1, int(round(len(req_skills) * random.uniform(0.8, 1.0))))
        cand_skills = random.sample(req_skills, k=k_req) + random.sample(opt_skills, k=min(len(opt_skills), random.randint(1, 2)))
    elif match_tier == "PARTIAL":
        # Candidate has 40%-70% of required skills
        k_req = max(1, int(round(len(req_skills) * random.uniform(0.4, 0.7))))
        cand_skills = random.sample(req_skills, k=k_req)
        # Add 1-2 extra skills from same domain
        extras = [s for s in cv_dom["core_skills"] if s not in cand_skills]
        cand_skills += random.sample(extras, k=min(2, len(extras)))
    elif match_tier == "EXP_GAP":
        # Good skill overlap (70%-90%) but lower experience years than Senior/Middle JD
        k_req = max(1, int(round(len(req_skills) * random.uniform(0.7, 0.9))))
        cand_skills = random.sample(req_skills, k=k_req)
    else:  # MISMATCH
        # Candidate is from a different domain with 0-1 overlapping skill
        cand_skills = random.sample(cv_dom["core_skills"], k=min(4, len(cv_dom["core_skills"])))

    cand_skills = list(dict.fromkeys(cand_skills))

    jd_text = (
        f"Vị trí tuyển dụng: {jd_title} ({jd_level[0]}).\n"
        f"Mô tả công việc: {jd_dom['jd_desc']}\n"
        f"Yêu cầu kỹ năng bắt buộc: {', '.join(req_skills)}.\n"
        f"Kỹ năng ưu tiên: {', '.join(opt_skills)}.\n"
        f"Yêu cầu kinh nghiệm: {jd_level[2]} (tối thiểu {jd_level[1]} năm)."
    )

    cv_headline = random.choice(cv_dom["jd_titles"])
    cv_text = (
        f"Hồ sơ ứng viên: {cv_headline} với {cv_years} năm kinh nghiệm thực tế.\n"
        f"Tóm tắt năng lực: Đã tham gia phát triển và vận hành các dự án thuộc lĩnh vực {cv_dom['domain']}. "
        f"{cv_dom['jd_desc'][:140]}.\n"
        f"Kỹ năng chuyên môn: {', '.join(cand_skills)}.\n"
        f"Kinh nghiệm làm việc: {cv_years} năm làm việc tại doanh nghiệp CNTT, trực tiếp sử dụng {', '.join(cand_skills[:3])} "
        f"để xây dựng sản phẩm, tối ưu hiệu năng và phối hợp nhóm theo quy trình Agile/Scrum."
    )

    # Compute ground-truth expert annotation score (0..100)
    req_set = {s.lower() for s in req_skills}
    opt_set = {s.lower() for s in opt_skills}
    cand_set = {s.lower() for s in cand_skills}

    matched_req = len(req_set & cand_set)
    matched_opt = len(opt_set & cand_set)
    skill_score = (matched_req / max(1, len(req_set))) * 88.0 + (matched_opt / max(1, len(opt_set))) * 12.0
    skill_score = min(100.0, max(10.0, skill_score))

    need_years = jd_level[1]
    if cv_years >= need_years:
        exp_score = min(100.0, 88.0 + min(12.0, (cv_years - need_years) * 3.5))
    elif need_years - cv_years == 1:
        exp_score = 70.0
    elif need_years - cv_years == 2:
        exp_score = 52.0
    else:
        exp_score = 35.0

    same_domain = 1.0 if jd_dom["domain"] == cv_dom["domain"] else 0.25
    keyword_score = min(100.0, max(15.0, same_domain * 60.0 + (matched_req / max(1, len(req_set))) * 40.0))

    ground_truth = round(0.60 * skill_score + 0.25 * exp_score + 0.15 * keyword_score, 1)
    ground_truth = min(98.0, max(12.0, ground_truth))

    return {
        "id": pair_id,
        "domain_jd": jd_dom["domain"],
        "domain_cv": cv_dom["domain"],
        "match_tier": match_tier,
        "jd_title": jd_title,
        "jd_exp_level": jd_level[0],
        "jd_required_years": need_years,
        "jd_required_skills": req_skills,
        "jd_optional_skills": opt_skills,
        "jd_text": jd_text,
        "cv_headline": cv_headline,
        "cv_years": cv_years,
        "cv_skills": cand_skills,
        "cv_text": cv_text,
        "skill_score": round(skill_score, 1),
        "exp_score": round(exp_score, 1),
        "keyword_score": round(keyword_score, 1),
        "ground_truth_score": ground_truth,
        "fit_label": 1 if ground_truth >= 70.0 else 0,
    }


def generate_all(output_path: Path, n_samples: int = 360):
    pairs = []
    pid = 1
    tiers = ["STRONG", "PARTIAL", "EXP_GAP", "MISMATCH"]
    weights = [0.35, 0.30, 0.15, 0.20]

    while len(pairs) < n_samples:
        tier = random.choices(tiers, weights=weights, k=1)[0]
        jd_dom = random.choice(DOMAINS)
        if tier == "MISMATCH":
            other_doms = [d for d in DOMAINS if d["domain"] != jd_dom["domain"]]
            cv_dom = random.choice(other_doms)
            jd_level = random.choice(EXP_LEVELS)
            cv_years = random.randint(0, 5)
        elif tier == "EXP_GAP":
            cv_dom = jd_dom
            jd_level = random.choice(EXP_LEVELS[3:])  # MIDDLE or SENIOR
            cv_years = random.randint(0, 1)
        elif tier == "STRONG":
            cv_dom = jd_dom
            jd_level = random.choice(EXP_LEVELS)
            cv_years = max(jd_level[1], jd_level[1] + random.randint(0, 3))
        else:  # PARTIAL
            cv_dom = jd_dom
            jd_level = random.choice(EXP_LEVELS)
            cv_years = max(0, jd_level[1] + random.randint(-1, 2))

        pairs.append(build_pair(pid, jd_dom, cv_dom, jd_level, cv_years, tier))
        pid += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(pairs)} labeled CV-JD pairs -> {output_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "dataset" / "cv_jd_vietnamese_dataset.json"
    generate_all(out, 360)
