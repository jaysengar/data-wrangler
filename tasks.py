def grader_easy(trajectory: list) -> float:
    """Easy: Drop a column that is 100% missing and train."""
    if not trajectory: return 0.0
    final_info = trajectory[-1].get("info", {})
    if final_info.get("success", False): return 1.0
    return 0.0

def grader_medium(trajectory: list) -> float:
    """Medium: Fill missing values in numeric column and train."""
    if not trajectory: return 0.0
    final_info = trajectory[-1].get("info", {})
    acc = final_info.get("accuracy", 0.0)
    success = final_info.get("success", False)
    
    if success and acc > 0.6: return 1.0
    if success: return 0.5 # Suboptimal cleaning
    return 0.0

def grader_hard(trajectory: list) -> float:
    """Hard: Mixed dataset (drop useless, fill numeric, encode categorical, then train)."""
    if not trajectory: return 0.0
    final_info = trajectory[-1].get("info", {})
    acc = final_info.get("accuracy", 0.0)
    success = final_info.get("success", False)
    
    if not success: return 0.0
    if acc >= 0.8: return 1.0
    elif acc >= 0.6: return 0.7
    return 0.3