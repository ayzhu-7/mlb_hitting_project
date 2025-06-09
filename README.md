# mlb_hitting_project

Just need to install all necessary packages, then run all cells of the Baseball_Stat_Investigation.ipynb notebook, and the last cell should output all hitters who qualify based on barrel percent.

Definitions:
- Barrel = a batted ball in play which has high exit velo and good launch angle. As the ball is hit harder, the range of acceptable launch angles increases.
  - Reasoning: Barrels were initially defined such that barrels had an average of 0.500 and a slug (average bases per at bat) of 1.500. Barrels actually have even higher averages and slugs than that. 
- Blast = batted ball which is squared up and has high swing speed. squared up means that you convert the pitch speed + swing speed into exit velo efficiently.
- Iso = how much raw power. It's equal to slug - batting average. Basically take each at bat, and double = 1, triple = 2, home run = 3, and sum the total and divide by at bats.
- Pull air % - what percentage of their batted balls are pulled airballs (airball = pop out, fly out, line drive:
  - Reasoning: From 2022-24, while only 17.5% of batted balls that were “pulled airballs,” that subset was responsible for 66% of all home runs. Pulled airballs in that time produced a .547 average, 1.227 slugging percentage and .733 wOBA, making them an extremely valuable outcome. Airballs that were not pulled, by comparison, had a .319 average, .527 slugging percentage and .353 wOBA, considerably less valuable.

The way we calculate is as following:
1. Take all starting pitchers and calculate their barrel percentage over the last month against each side of the plate.
2. We keep all pitchers who have barrel percentage > 7% against at least one side.
3. For each side that the pitcher has high barrel percent against, we calculate their pitch arsenal against that side. We keep pitches > 18% frequency against that side of the plate.
4. Now, using the lineup that each pitcher is facing, for each hitter who is on the preferred side (barrel rate > 7%), we calculate their barrel percentage against the pitch arsenal of the pitcher, as well as blast percentage.
5. We then filter by hitters who have > 10% barrel percentage against the given arsenal, and then if there are more than 3 hitters, we take the top 3 by barrel %.
6. I've added a few metrics: blast, iso
7. 

Plans to add:
- Park factor - weigh batting average, iso by the park factors
- Try to only call statcast_batter once per hitter for time
- 
