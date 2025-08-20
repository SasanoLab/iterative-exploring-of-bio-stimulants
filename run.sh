exp_dir=$PROJECT_DIR
cd $exp_dir
. "$exp_dir/.envrc"

log_file="$exp_dir/cron.log"
cmd="$PROJECT_DIR/.rye/shims/rye run"

touch $log_file
last_line=$(/usr/bin/tail -n 1 $log_file)

pid_self=$$
pid_oldest=$(/usr/bin/pgrep -fo $exp_dir/run.sh)

# 他に実行中のcron jobがないか確認
# pid_selfがこのスクリプトのpid, pid_oldestがcron自体のpidなので1つ差がある
# pid_oldestが空でない場合、他のcron jobが実行中
# pid_oldestが空の場合、cron jobが動いていない
if [ $pid_self -ne $(expr $pid_oldest + 1) ] && [ ! -z $pid_oldest ]; then
    echo "Another cron job is running. Exit this job. ($pid_oldest) ($pid_self)"
    exit 0
fi

# 監視対象ファイルが更新されているか確認
# 行数が同じなら1, 違うなら0
# 行数が違う場合(こちらの推薦化合物の方が量が多い場合)、シミュレーションがまだ行われておらず、更新は不要
$cmd python src/misc/check_update.py
if [ $? -eq 0 ]; then
    msg="No updates required"
    echo $last_line | /usr/bin/grep "$msg" >/dev/null
    if [ $? -eq 1 ]; then
        timestamp=$(/usr/bin/date "+%Y-%m-%d/%H-%M-%S")
        echo "$timestamp: $msg" >>$log_file
    fi
    exit 0
fi

timestamp=$(/usr/bin/date "+%Y-%m-%d/%H-%M-%S")
echo "$timestamp: Simulation is done. Update models!" >>$log_file

$cmd python src/misc/background_work.py

# スコアリングモデルの更新
$cmd accelerate launch --config_file accelerate.json src/docking/train.py

if [ $? -ne 0 ]; then
    timestamp=$(/usr/bin/date "+%Y-%m-%d/%H-%M-%S")
    echo "$timestamp: src/docking/train.py is faield" >>$log_file
    exit 1
fi

timestamp=$(/usr/bin/date "+%Y-%m-%d/%H-%M-%S")
echo "$timestamp: Models are updated. Updating recommendations..." >>$log_file

# スコアリングモデルを用いて推薦化合物を決定、isometric_candidates.txtを更新
$cmd python src/recommend.py

if [ $? -ne 0 ]; then
    timestamp=$(/usr/bin/date "+%Y-%m-%d/%H-%M-%S")
    echo "$timestamp: src/recommend.py is faield" >>$log_file
    exit 1
fi

# 化合物選定モデルの評価
$cmd python src/eval.py
if [ $? -ne 0 ]; then
    timestamp=$(/usr/bin/date "+%Y-%m-%d/%H-%M-%S")
    echo "$timestamp: src/eval.py is faield" >>$log_file
    exit 1
fi

timestamp=$(/usr/bin/date "+%Y-%m-%d/%H-%M-%S")
echo "$timestamp: Models, recommendations, and evaluations are updated successfuly!" >>$log_file
