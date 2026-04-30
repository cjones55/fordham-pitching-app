def add_contact_quality_local(batted_balls):
    # Exclude fouls and bunts from EV calculations
    batted_balls = [bb for bb in batted_balls if bb['type'] not in ['foul', 'bunt']]
    # Compute hard hit and barrel flags
    for bb in batted_balls:
        bb['hard_hit'] = calculate_hard_hit(bb)
        bb['barrel'] = calculate_barrel(bb)
    return batted_balls
