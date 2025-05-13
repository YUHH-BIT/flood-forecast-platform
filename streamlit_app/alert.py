# alert.py
import datetime

def run_alert_module(prediction_data):
    """
    运行预警模块，根据预测的水文数据判断是否触发预警。
    :param prediction_data: 预测的洪水数据（DataFrame 格式）
    :return: 预警信息或结果
    """
    alert_messages = []
    
    for index, row in prediction_data.iterrows():
        predicted_runoff = row['predicted_runoff']
        date = row['date']

        # 设定一个假定的阈值
        if predicted_runoff > 100:  # 假设 100 mm 为洪水阈值
            alert_messages.append(f"🚨【预警】{date} 预测的径流量 {predicted_runoff} mm 超过阈值，存在洪水风险！")
        else:
            alert_messages.append(f"✅ {date} 预测的径流量 {predicted_runoff} mm 正常，无洪水风险。")
    
    return alert_messages

