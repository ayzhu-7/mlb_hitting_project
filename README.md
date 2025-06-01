# mlb_hitting_project

Just need to install all necessary packages, then run all cells of the Baseball_Stat_Investigation.ipynb notebook, and the last cell should output all hitters who qualify based on barrel percent.

Definitions:
Barrel = a batted ball in play which has high exit velo and good launch angle. As the ball is hit harder, the range of acceptable launch angles increases.
Blast = batted ball which is squared up and has high swing speed. squared up means that you convert the pitch speed + swing speed into exit velo efficiently.

The way we calculate is as following:
1. Take all starting pitchers and calculate their barrel percentage over the last month against each side of the plate.
2. We keep all pitchers who have barrel percentage > 7% against at least one side.
3. For each side that the pitcher has high barrel percent against, we calculate their pitch arsenal against that side. We keep pitches > 18% frequency against that side of the plate.
4. Now, using the lineup that each pitcher is facing, for each hitter who is on the preferred side (barrel rate > 7%), we calculate their barrel percentage against the pitch arsenal of the pitcher, as well as blast percentage.
5. We then filter by hitters who have > 10% barrel percentage against the given arsenal, and then if there are more than 3 hitters, we take the top 3 by barrel %. 
