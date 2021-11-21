#!/usr/bin/env ruby

# extract-timelog-all.rb
# 指定されたバス停区間に存在する、全バス停間の運行記録を取り出す

require 'sqlite3'
require 'date'
require 'holiday_japan'

def main
  # DBに接続
  client = SQLite3::Database.new("timelog.db")

  # 登録されているASL-IDと、各ASL-IDのシーケンス番号最大値を求める
  asl_id_query = client.query("SELECT ASL_ID, MAX(seq_no) FROM timelog GROUP BY ASL_ID;")
  asl_id_list  = []
  asl_id_query.each do |row|
    asl_id_list << { :asl_id => row[0], :max_seq_no => row[1] }
  end

  query = %q{SELECT ASL_ID, seq_no, via, BSCode, ArrivalTime, DepartureTime
             FROM timelog
             WHERE ASL_ID = ? AND seq_no = ?
            }
  statement = client.prepare(query)

  asl_id_list.each do |id_and_max_seq|
    1.upto(id_and_max_seq[:max_seq_no]) do |i|
      # 各シーケンスを取り出す
      result = statement.execute(id_and_max_seq[:asl_id], i)

      # ループ外に保存する変数群
      sbscode = nil
      sbs_departuretime = nil
      bus_operation_id  = nil
      departure_time_at_sbs_on_time = nil

      # 1シーケンスの各行について繰り返す
      result.each_with_index do |row, idx|
        # 時刻情報が無ければ次の行へ
        next if row[5] == ""

        # 起点停留所がまだ検出されていなければ
        if sbscode.nil?
          # 起点停留所IDを登録する
          sbscode = row[3].to_i

          # 起点停留所出発時刻を登録する
          sbs_departuretime = Time.parse(row[5])

          # 時刻表と照合して便番号を求める
          bus_operation_id = detect_operation_id(row[3], sbs_departuretime)

          # 起点停留所の出発時刻を求める
          # もしそのような便と停留所の時刻組合せが存在しなかった場合は
          # 次の行へ進む（回送便などの場合が該当）
          result_timetable = search_timetable(bus_operation_id, sbscode)
          if result_timetable.nil?
            sbscode = nil
            next
          end

          # 起点停留所の時刻表上の出発時刻を、加減算可能なようにゲタを履かせる
          departure_time_at_sbs_on_time = Time.parse(sbs_departuretime.strftime("%Y-%m-%d") + " " + result_timetable + ":00")
        else
          # 起点停留所が検出済みの場合
          # 検出した停留所の着発時刻を求める
          fbs_arrivaltime   = Time.parse(row[4])
          fbs_departuretime = Time.parse(row[5])

          # 便番号が求められていない場合は求める
          bus_operation_id = detect_operation_id(row[3], fbs_departuretime) if bus_operation_id.nil?

          # 検出した停留所の出発時刻を求める
          result_timetable = search_timetable(bus_operation_id, row[3])

          # もしそのような便と停留所の組合せが存在しなかった場合は
          # 次の行へ進む
          if result_timetable.nil?
            next
          else
            # 検出した停留所の時刻表上の出発時刻を求める
            departure_on_time = Time.parse(fbs_departuretime.strftime("%Y-%m-%d") + " " + result_timetable + ":00")

            # 到着時遅延秒数、出発時遅延秒数を求める
            diff_arrival = fbs_arrivaltime - departure_on_time
            diff_departure = fbs_departuretime - departure_on_time

            # 情報を出力する
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                    sbs_departuretime.strftime('%Y-%m-%d'), # 運行日
                    row[0],                                 # ASL-ID
                    row[1],                                 # シーケンス番号
                    row[2],                                 # 経由
                    bus_operation_id,                       # 便番号
                    sbscode,                                # 起点停留所ID
                    departure_time_at_sbs_on_time.strftime('%H:%M:%S'),  # 起点定時
                    sbs_departuretime.strftime('%H:%M:%S'), # 起点出発時刻
                    row[3],                                 # 着点停留所ID
                    departure_on_time.strftime('%H:%M:%S'), # 着点定時
                    fbs_arrivaltime.strftime('%H:%M:%S'),   # 着点到着時刻
                    fbs_departuretime.strftime('%H:%M:%S'), # 着点出発時刻
                    diff_arrival,                           # 到着時遅延秒数
                    diff_departure                          # 出発時遅延秒数
          end
        end
      end
    end
  end
  client.close
end

# detect_operation_id
# 停留所IDと出発時刻の組から、便番号をひとつ決定する
def detect_operation_id(sbscode, sbs_departuretime)
  timetable_db = SQLite3::Database.new("timetable/timetable.db")
  query = %q{
    SELECT OperationID FROM Timetable
    WHERE BusStopID = ?
    AND DepartTime >= ?
    AND DepartTime <= ?
    ORDER BY OperationID ASC
  }
  query_straddling_day = %q{
    SELECT OperationID FROM Timetable
    WHERE BusStopID = ?
    AND ((DepartTime >= ? AND DepartTime <= '23:59')
    OR (DepartTime >= '00:00' AND DepartTime <= ?))
    ORDER BY OperationID ASC 
  }

  # 時刻表上の出発時刻から2分以内に出発する便を、当該便と認める
  sbs_departuretime_start = (sbs_departuretime - 120).strftime('%H:%M')
  sbs_departuretime_end   = sbs_departuretime.strftime('%H:%M')

  # 0時またがりで判定する場合は、クエリを変更する
  if (sbs_departuretime - 120).day != sbs_departuretime.day
    result = timetable_db.execute(query_straddling_day, [sbscode, sbs_departuretime_start, sbs_departuretime_end])
  else
    result = timetable_db.execute(query, [sbscode, sbs_departuretime_start, sbs_departuretime_end])
  end
  result.flatten!

  # 日付曜日によりどのダイヤに対応するかを絞る
  case sbs_departuretime.to_date
  when Date.parse('2019-04-01') .. Date.parse('2019-09-30')
    timetable_begindate = "20190401"
  when Date.parse('2019-10-01') .. Date.parse('2020-03-31')
    timetable_begindate = "20191001"
  when Date.parse('2020-04-01') .. Date.parse('2020-09-30')
    timetable_begindate = "20200401"
  when Date.parse('2020-10-01') .. Date.parse('2021-03-31')
    timetable_begindate = "20201001"
  when Date.parse('2021-04-01') .. Date.parse('2021-09-30')
    timetable_begindate = "20210401"
  else
    return nil
  end

  if HolidayJapan.check(sbs_departuretime) or sbs_departuretime.sunday?
    dia_day = "H"
  elsif sbs_departuretime.saturday?
    dia_day = "S"
  else
    dia_day = "W"
  end

  operation_id = result.select {|ids| ids.include?("B" + timetable_begindate + "-" + dia_day)}.shift
  
  timetable_db.close

  operation_id
end

def search_timetable(bus_operation_id, busstop_id)
  timetable_db = SQLite3::Database.new("timetable/timetable.db")
  query = %q{
    SELECT DepartTime FROM Timetable
    WHERE OperationID = ?
    AND BusStopID = ?
  }
  result = timetable_db.execute(query, [bus_operation_id, busstop_id])
  timetable_db.close

  result.flatten.shift
end

if $0 == __FILE__
  main
end

