from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def process_sdf(sdf_drive, sdf_vehicle):
  sdf_join_drive_vehicle = sdf_drive.alias("drive").join(sdf_vehicle.alias("vehicle"), ["vehicle_id"])
  sdf_join_drive_vehicle_fillna = sdf_join_drive_vehicle.fillna(0)
  sdf_drive_start_of_week = sdf_join_drive_vehicle_fillna.withColumn("week_start_date", \
                                                                     (F.date_sub(F.next_day(
                                                                       F.from_utc_timestamp(F.col("datetime"),
                                                                                            "America/New_York"),
                                                                       'monday'), 7)))

  sdf_Active_horsepower =  sdf_drive_start_of_week.withColumn("Active_horsepower" ,  (F.col("eng_load") / 255) \
                                                              * (F.col("max_torque") * F.col("rpm"))  / 5252)

  # Horsepower utilization – Active horsepower / Max Horsepower
  sdf_Horsepower_utilization = sdf_Active_horsepower.withColumn("Horsepower_utilization", F.col("Active_horsepower") / F.col("max_horsepower"))

  # # Torque Utilization - calculated as Engine load/ 255
  sdf_Torque_Utilization = sdf_Horsepower_utilization.withColumn("Torque_Utilization", F.col("eng_load") / 255)

  # # RPM Utilization – RPM / Maximum horsepower rpm
  sdf_RPM_Utilization = sdf_Torque_Utilization.withColumn("RPM_Utilization", F.col("rpm") / F.col("max_horsepower_rpm") )

  sdf_engine_features = sdf_RPM_Utilization.withColumn("ft_torque_util_60pct_s",
                                                       F.when((F.col("Torque_Utilization") >= 0.6) \
                                                              & (F.col("Torque_Utilization") < 0.7), \
                                                              F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_torque_util_70pct_s", F.when((F.col("Torque_Utilization") >= 0.7) \
                                                 & (F.col("Torque_Utilization") < 0.8), \
                                                 F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_torque_util_80pct_s", F.when((F.col("Torque_Utilization") >= 0.8) \
                                                 & (F.col("Torque_Utilization") < 0.9), \
                                                 F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_torque_util_90pct_s", F.when((F.col("Torque_Utilization") >= 0.9) \
                                                 & (F.col("Torque_Utilization") < 1), \
                                                 F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_horsepower_util_50pct_s", F.when((F.col("Horsepower_utilization") >= 0.5) \
                                                     & (F.col("Horsepower_utilization") < 0.6), \
                                                     F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_horsepower_util_60pct_s", F.when((F.col("Horsepower_utilization") >= 0.6) \
                                                     & (F.col("Horsepower_utilization") < 0.7), \
                                                     F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_horsepower_util_70pct_s", F.when((F.col("Horsepower_utilization") >= 0.7) \
                                                     & (F.col("Horsepower_utilization") < 0.8), \
                                                     F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_horsepower_util_80pct_s", F.when((F.col("Horsepower_utilization") >= 0.8) \
                                                     & (F.col("Horsepower_utilization") < 0.9), \
                                                     F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_rpm_util_50pct_s", F.when((F.col("RPM_Utilization") >= 0.5) \
                                              & (F.col("RPM_Utilization") < 0.6), \
                                              F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("ft_rpm_util_60pct_s", F.when((F.col("RPM_Utilization") >= 0.6) \
                                              & (F.col("RPM_Utilization") < 0.7), \
                                              F.lit(1)).otherwise(F.lit(0))) \
 \
          sdf_engine_features_total = sdf_engine_features.select("vehicle_id", "week_start_date", "datetime", \
                                                                 "ft_torque_util_60pct_s", "ft_torque_util_70pct_s",
                                                                 "ft_torque_util_80pct_s", "ft_torque_util_90pct_s", \
                                                                 "ft_horsepower_util_50pct_s",
                                                                 "ft_horsepower_util_60pct_s",
                                                                 "ft_horsepower_util_70pct_s",
                                                                 "ft_horsepower_util_80pct_s", \
                                                                 "ft_rpm_util_50pct_s", "ft_rpm_util_60pct_s")

  sdf_sdf_engine_features_agg = sdf_engine_features_total.groupBy("vehicle_id", "week_start_date") \
    .agg(F.sum("ft_torque_util_60pct_s").alias("ft_torque_util_60pct_s"), \
         F.sum("ft_torque_util_70pct_s").alias("ft_torque_util_70pct_s"), \
         F.sum("ft_torque_util_80pct_s").alias("ft_torque_util_80pct_s"), \
         F.sum("ft_torque_util_90pct_s").alias("ft_torque_util_90pct_s"), \
         F.sum("ft_horsepower_util_50pct_s").alias("ft_horsepower_util_50pct_s"), \
         F.min("ft_horsepower_util_60pct_s").alias("ft_horsepower_util_60pct_s"), \
         F.min("ft_horsepower_util_70pct_s").alias("ft_horsepower_util_70pct_s"), \
         F.min("ft_horsepower_util_80pct_s").alias("ft_horsepower_util_80pct_s"), \
         F.min("ft_rpm_util_50pct_s").alias("ft_rpm_util_50pct_s"), \
         F.min("ft_rpm_util_60pct_s").alias("ft_rpm_util_60pct_s"), )

  sdf_sdf_engine_features_final = sdf_sdf_engine_features_agg.select("vehicle_id", "week_start_date", \
                                                                     "ft_torque_util_60pct_s", "ft_torque_util_70pct_s",
                                                                     "ft_torque_util_80pct_s", "ft_torque_util_90pct_s", \
                                                                     "ft_horsepower_util_50pct_s",
                                                                     "ft_horsepower_util_60pct_s",
                                                                     "ft_horsepower_util_70pct_s",
                                                                     "ft_horsepower_util_80pct_s", \
                                                                     "ft_rpm_util_50pct_s", "ft_rpm_util_60pct_s")

  sdf_sdf_engine_features_final = sdf_sdf_engine_features_final.sort(F.col("vehicle_id"), F.col("week_start_date"))

  sdf_sdf_engine_features_final = sdf_sdf_engine_features_final.withColumn("week_start_date",
                                                                           F.date_format(F.col("week_start_date"),
                                                                                         "yyyy-MM-dd"))

  sdf_sdf_engine_features_final = sdf_sdf_engine_features_final.fillna(0)

  return sdf_sdf_engine_features_final