def pytest_sessionfinish(session, exitstatus):
    # An empty test tree is a valid state during INIT (see TASKS.md INIT-1) — pytest's
    # default exit code 5 ("no tests collected") would otherwise fail CI on a clean bootstrap.
    if exitstatus == 5:
        session.exitstatus = 0
