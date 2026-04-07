def grader_easy(trajectory: list) -> float:
    """Easy: Drop a column that is 100% missing and train."""
    if not trajectory: return 0.01
    final_info = trajectory[-1].get("info", {})
    if final_info.get("success", False): return 0.98
    return 0.01

def grader_medium(trajectory: list) -> float:
    """Medium: Fill missing values in numeric column and train."""
    if not trajectory: return 0.01
    final_info = trajectory[-1].get("info", {})
    acc = final_info.get("accuracy", 0.0)
    success = final_info.get("success", False)
    
    if success and acc > 0.6: return 0.98
    if success: return 0.51 # Suboptimal cleaning
    return 0.01

def grader_hard(trajectory: list) -> float:
    """Hard: Mixed dataset (drop useless, fill numeric, encode categorical, then train)."""
    if not trajectory: return 0.01
    final_info = trajectory[-1].get("info", {})
    acc = final_info.get("accuracy", 0.0)
    success = final_info.get("success", False)
    
    if not success: return 0.01
    if acc >= 0.8: return 0.98
    elif acc >= 0.6: return 0.71
    return 0.31