/****** Script for SelectTopNRows command from SSMS  ******/
-- 區域載客數統計總量
SELECT Zone_ID,sum(Hire_count) 人數總計
FROM [taxi].[dbo].[train_hire_stats]
group by Zone_ID
order by Zone_ID
/*---------------------------------------------------------------------*/
-- 區域時間載客數
SELECT Zone_ID,Date,sum(Hire_count) 載客數
FROM [taxi].[dbo].[train_hire_stats]
group by Zone_ID,Date
order by Zone_ID,Date