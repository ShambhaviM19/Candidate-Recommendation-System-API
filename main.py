from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from fuzzywuzzy import fuzz
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import re

app = FastAPI()

geolocator = Nominatim(user_agent="candidate_recommender")

class Education(BaseModel):
    degree: str
    specialization: str
    institute: str
    start_year: int
    end_year: int
    start_month: str
    end_month: str
    isStudent: bool

class Experience(BaseModel):
    company_name: str
    designation: str
    description: str
    start_year: int
    end_year: int
    start_month: str
    end_month: str
    isWorking: bool

class Project(BaseModel):
    project_name: str
    project_description: str

class Candidate(BaseModel):
    email: str
    name: str
    number: str
    skills: List[str]
    current_location: str
    preferred_location: str
    current_salary: float
    expected_salary: float
    open_to_work: bool
    summary: str
    notice_period: int
    d_o_b: str
    total_experience: float
    linkedin_link: str
    current_company_name: str
    github_id: str
    education: List[Education]
    experience: List[Experience]
    certification: List[str]
    projects: List[Project]

class Job(BaseModel):
    job_title: str
    job_role: str
    work_mode: str
    skills: List[str]
    employment_type: str
    company_name: str
    location: str
    experience: str
    salary_start: float
    salary_end: float
    notice_period: int
    preferred_degree: str
    industry_type: str
    job_description: str

class RecommendationRequest(BaseModel):
    jd: Job
    candidates: List[Candidate]

def calculate_location_scores(job_location: str, candidate_locations: List[str]) -> Dict[str, float]:
    distances = {}
    total_distance = 0
    job_coords = geolocator.geocode(job_location)
    
    for location in candidate_locations:
        try:
            candidate_coords = geolocator.geocode(location)
            
            if job_coords and candidate_coords:
                distance = geodesic((job_coords.latitude, job_coords.longitude), 
                                    (candidate_coords.latitude, candidate_coords.longitude)).kilometers
                distances[location] = distance
                total_distance += distance
            else:
                distances[location] = float('inf')
        except:
            distances[location] = float('inf')
    
    scores = {}
    for location, distance in distances.items():
        if distance == float('inf'):
            scores[location] = 0
        else:
            scores[location] = 1 - (distance / total_distance)
    
    return scores

def parse_experience(experience_str: str) -> tuple:
    match = re.search(r'(\d+)\s*-\s*(\d+)', experience_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    single_match = re.search(r'(\d+)', experience_str)
    if single_match:
        return int(single_match.group(1)), int(single_match.group(1))
    return 0, 0

def calculate_experience_score(candidate_experience: float, job_experience: str) -> float:
    min_exp, max_exp = parse_experience(job_experience)
    if min_exp <= candidate_experience <= max_exp:
        return 1.0
    elif candidate_experience < min_exp:
        return max(0, 1 - (min_exp - candidate_experience) / min_exp)
    else:
        return max(0, 1 - (candidate_experience - max_exp) / max_exp)

def calculate_skill_score(candidate_skills: List[str], job_skills: List[str]) -> float:
    if not candidate_skills or not job_skills:
        return 0
    
    matched_scores = []
    for job_skill in job_skills:
        best_match = max(fuzz.token_set_ratio(job_skill.lower(), candidate_skill.lower()) for candidate_skill in candidate_skills)
        matched_scores.append(best_match / 100)
    
    return sum(matched_scores) / len(job_skills)

def parse_salary(salary_str: str) -> tuple:
    match = re.findall(r'(\d+),?(\d+),?(\d+)', salary_str)
    if match:
        return float(''.join(match[0])), float(''.join(match[-1]))
    return 0, 0

def calculate_salary_score(job_salary: str, candidate_expected_salary: Optional[float]) -> float:
    salary_min, salary_max = map(float, job_salary.split('-'))
    
    if candidate_expected_salary is None:
        return 1
    if salary_min <= candidate_expected_salary <= salary_max:
        return 1
    elif candidate_expected_salary < salary_min:
        return max(0, 1 - (salary_min - candidate_expected_salary) / salary_min)
    else:
        return max(0, 1 - (candidate_expected_salary - salary_max) / salary_max)
    
def calculate_notice_period_score(candidate_notice_period: Optional[int], job_required_joining_time: Optional[int]) -> float:
    if candidate_notice_period is None or job_required_joining_time is None:
        return 1
    
    if candidate_notice_period <= job_required_joining_time:
        return 1
    
    difference = abs(candidate_notice_period - job_required_joining_time)
    max_difference = max(candidate_notice_period, job_required_joining_time)
    
    return max(0, 1 - (difference / max_difference))

def recommend_candidates(job: Job, candidates: List[Candidate], weights: Dict[str, float]) -> List[tuple]:
    job_skills = job.skills
    job_experience = job.experience
    job_location = job.location
    job_salary = f"{job.salary_start} - {job.salary_end}"  
    
    location_scores = calculate_location_scores(job_location, [candidate.current_location for candidate in candidates])
    
    candidate_scores = []
    
    for candidate in candidates:
        skill_score = calculate_skill_score(candidate.skills, job_skills)
        experience_score = calculate_experience_score(candidate.total_experience, job_experience)
        location_score = location_scores[candidate.current_location]
        salary_score = calculate_salary_score(job_salary, candidate.expected_salary)
        notice_period_score = calculate_notice_period_score(candidate.notice_period, job.notice_period)
        
        hybrid_score = (
            weights['skills'] * skill_score +
            weights['experience'] * experience_score +
            weights['location'] * location_score +
            weights['salary'] * salary_score +
            weights['notice_period'] * notice_period_score
        )
        
        candidate_scores.append((candidate, hybrid_score, skill_score, experience_score, location_score, salary_score, notice_period_score))
    
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    return candidate_scores

@app.post("/recommend_candidates/")
async def recommend_candidates_endpoint(request: RecommendationRequest):
    weights = {
        'skills': 0.7,
        'experience': 0.15,
        'location': 0.05,
        'salary': 0.05,
        'notice_period': 0.05
    }
    
    job = request.jd
    candidates = request.candidates
    
    recommended_candidates = recommend_candidates(job, candidates, weights)

    results = []
    for candidate, hybrid_score, skill_score, experience_score, location_score, salary_score, notice_period_score in recommended_candidates:
        matched_skills = [skill.lower() for skill in candidate.skills if skill.lower() in [s.lower() for s in job.skills]]
        unmatched_skills = [skill for skill in job.skills if skill.lower() not in [s.lower() for s in matched_skills]]

        result = {
            "name": candidate.name,
            "email": candidate.email,
            "match": {
                "score": round(hybrid_score * 100, 1),
                "matching_items": {
                    "location": "Yes" if location_score > 0.5 else "No",
                    "experience": "Yes" if experience_score > 0.5 else "No",
                    "salary": "Yes" if salary_score > 0.5 else "No",
                    "notice_period": "Yes" if notice_period_score > 0.5 else "No",
                    "skills": "Full Match" if len(unmatched_skills) == 0 else "Partial Match",
                    "matched_skills": matched_skills,
                    "unmatched_skills": unmatched_skills
                }
            }
        }
        results.append(result)

    return {
        "results": results,
        "total_candidates": len(candidates)
    }