import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


MISSING = "MISSING"
DATE_FMT = "%Y-%m-%d"


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_date_parts(value):
    try:
        dt = datetime.strptime(value, DATE_FMT)
        return dt.year, dt.month, dt.toordinal()
    except (TypeError, ValueError):
        return 0, 0, None


def normalize_cat(value):
    if value is None or value == "":
        return MISSING
    return value


def count_multi(value):
    if value is None or value == "":
        return 0
    return value.count("|") + 1


def duration_bucket(duration_ms):
    if duration_ms < 0:
        return "missing"
    if duration_ms < 120000:
        return "<2m"
    if duration_ms < 180000:
        return "2-3m"
    if duration_ms < 240000:
        return "3-4m"
    if duration_ms < 300000:
        return "4-5m"
    if duration_ms < 600000:
        return "5-10m"
    return "10m+"


def age_bucket(age):
    if age == 0:
        return "0"
    if age < 0:
        return "missing"
    if age <= 18:
        return "1-18"
    if age <= 25:
        return "19-25"
    if age <= 35:
        return "26-35"
    if age <= 50:
        return "36-50"
    if age <= 100:
        return "51-100"
    return "100+"


def stable_fold_key(parts, n_splits):
    joined = "||".join(parts).encode("utf-8")
    digest = hashlib.md5(joined).hexdigest()
    return int(digest[:8], 16) % n_splits


def load_user_features(path):
    user_map = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            age = safe_int(row["age"], default=-1)
            rg_year, rg_month, rg_ordinal = safe_date_parts(row["rg_date"])
            exp_year, exp_month, exp_ordinal = safe_date_parts(row["exp_date"])
            membership_days = (
                exp_ordinal - rg_ordinal
                if rg_ordinal is not None and exp_ordinal is not None
                else 0
            )
            user_map[row["uid"]] = {
                "address": normalize_cat(row["address"]),
                "sex": normalize_cat(row["sex"]),
                "rg_source": normalize_cat(row["rg_source"]),
                "age": age,
                "age_bucket": age_bucket(age),
                "rg_year": rg_year,
                "rg_month": rg_month,
                "exp_year": exp_year,
                "exp_month": exp_month,
                "membership_days": membership_days,
                "is_age_zero": int(age == 0),
                "is_age_outlier": int(age < 0 or age > 100),
                "is_membership_days_negative": int(membership_days < 0),
            }
    return user_map


def collect_needed_episode_ids(train_path, test_path):
    needed = set()
    for path in (train_path, test_path):
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                needed.add(row["episode_id"])
    return needed


def load_episode_side_features(feature_path, additional_path, needed_episode_ids):
    episode_map = {}
    with feature_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episode_id = row["episode_id"]
            if episode_id not in needed_episode_ids:
                continue
            duration_ms = safe_int(row["duration"], default=-1)
            episode_map[episode_id] = {
                "language": normalize_cat(row["language"]),
                "duration_ms": duration_ms,
                "duration_sec": round(duration_ms / 1000.0, 3) if duration_ms >= 0 else -1,
                "duration_log1p": round(math.log1p(max(duration_ms, 0)), 6),
                "duration_bucket": duration_bucket(duration_ms),
                "category_count": count_multi(row["category_ids"]),
                "host_count": count_multi(row["host"]),
                "producer_count": count_multi(row["producer"]),
                "writer_count": count_multi(row["writer"]),
            }

    uuid_counter = Counter()
    add_map = {}
    with additional_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episode_id = row["episode_id"]
            if episode_id not in needed_episode_ids:
                continue
            uuid = normalize_cat(row["uuid"])
            title = row["title"] or ""
            uuid_counter[uuid] += 1
            add_map[episode_id] = {
                "title_token_count": count_multi(title),
                "title_char_len": len(title),
                "uuid": uuid,
            }

    for episode_id in needed_episode_ids:
        base = episode_map.setdefault(
            episode_id,
            {
                "language": MISSING,
                "duration_ms": -1,
                "duration_sec": -1,
                "duration_log1p": 0.0,
                "duration_bucket": "missing",
                "category_count": 0,
                "host_count": 0,
                "producer_count": 0,
                "writer_count": 0,
            },
        )
        extra = add_map.get(
            episode_id,
            {"title_token_count": 0, "title_char_len": 0, "uuid": MISSING},
        )
        base.update(extra)
        base["uuid_episode_count"] = uuid_counter.get(base["uuid"], 0)

    return episode_map


def add_freq(counter_map, key):
    counter_map[key] += 1


def add_target_stat(total_map, fold_map, key, fold_id, label):
    total_map[key][0] += 1
    total_map[key][1] += label
    fold_map[(key, fold_id)][0] += 1
    fold_map[(key, fold_id)][1] += label


def get_oof_te(total_map, fold_map, key, fold_id, global_mean):
    total_n, total_s = total_map.get(key, (0, 0))
    fold_n, fold_s = fold_map.get((key, fold_id), (0, 0))
    remain_n = total_n - fold_n
    remain_s = total_s - fold_s
    if remain_n <= 0:
        return global_mean, 0
    return remain_s / remain_n, remain_n


def get_test_te(total_map, key, global_mean):
    total_n, total_s = total_map.get(key, (0, 0))
    if total_n <= 0:
        return global_mean, 0
    return total_s / total_n, total_n


def build_feature_row(base_row, user_map, episode_map):
    uid = base_row["uid"]
    episode_id = base_row["episode_id"]
    user_feat = user_map.get(
        uid,
        {
            "address": MISSING,
            "sex": MISSING,
            "rg_source": MISSING,
            "age": -1,
            "age_bucket": "missing",
            "rg_year": 0,
            "rg_month": 0,
            "exp_year": 0,
            "exp_month": 0,
            "membership_days": 0,
            "is_age_zero": 0,
            "is_age_outlier": 1,
            "is_membership_days_negative": 0,
        },
    )
    episode_feat = episode_map.get(
        episode_id,
        {
            "language": MISSING,
            "duration_ms": -1,
            "duration_sec": -1,
            "duration_log1p": 0.0,
            "duration_bucket": "missing",
            "category_count": 0,
            "host_count": 0,
            "producer_count": 0,
            "writer_count": 0,
            "title_token_count": 0,
            "title_char_len": 0,
            "uuid": MISSING,
            "uuid_episode_count": 0,
        },
    )
    tab_name = normalize_cat(base_row["tab_name"])
    scene_name = normalize_cat(base_row["scene_name"])
    entrance_type = normalize_cat(base_row["entrance_type"])
    row = {
        "uid": uid,
        "episode_id": episode_id,
        "tab_name": tab_name,
        "scene_name": scene_name,
        "entrance_type": entrance_type,
    }
    row.update(user_feat)
    row.update(episode_feat)
    return row


def feature_fields():
    base_cats = [
        "uid",
        "episode_id",
        "tab_name",
        "scene_name",
        "entrance_type",
        "address",
        "sex",
        "rg_source",
        "age_bucket",
        "language",
        "duration_bucket",
        "uuid",
    ]
    base_nums = [
        "age",
        "rg_year",
        "rg_month",
        "exp_year",
        "exp_month",
        "membership_days",
        "is_age_zero",
        "is_age_outlier",
        "is_membership_days_negative",
        "duration_ms",
        "duration_sec",
        "duration_log1p",
        "category_count",
        "host_count",
        "producer_count",
        "writer_count",
        "title_token_count",
        "title_char_len",
        "uuid_episode_count",
    ]
    freq_fields = [
        "uid_count",
        "episode_count",
        "tab_name_count",
        "scene_name_count",
        "entrance_type_count",
        "uid_tab_count",
        "uid_entrance_count",
        "episode_tab_count",
        "episode_entrance_count",
    ]
    te_fields = [
        "te_uid",
        "te_episode_id",
        "te_tab_name",
        "te_scene_name",
        "te_entrance_type",
        "te_uid_tab",
        "te_uid_entrance",
        "te_episode_tab",
        "te_episode_entrance",
        "te_uid_train_count",
        "te_episode_id_train_count",
        "te_tab_name_train_count",
        "te_scene_name_train_count",
        "te_entrance_type_train_count",
        "te_uid_tab_train_count",
        "te_uid_entrance_train_count",
        "te_episode_tab_train_count",
        "te_episode_entrance_train_count",
        "global_label_mean",
    ]
    return base_cats + base_nums + freq_fields + te_fields


def build_stats(train_path, user_map, episode_map, n_splits):
    freq = {
        "uid": Counter(),
        "episode_id": Counter(),
        "tab_name": Counter(),
        "scene_name": Counter(),
        "entrance_type": Counter(),
        "uid_tab": Counter(),
        "uid_entrance": Counter(),
        "episode_tab": Counter(),
        "episode_entrance": Counter(),
    }
    te_total = {k: defaultdict(lambda: [0, 0]) for k in freq}
    te_fold = {k: defaultdict(lambda: [0, 0]) for k in freq}
    label_sum = 0
    row_count = 0

    with train_path.open("r", encoding="utf-8") as f:
        for row_idx, raw_row in enumerate(csv.DictReader(f)):
            label = int(raw_row["label"])
            row_count += 1
            label_sum += label
            feat_row = build_feature_row(raw_row, user_map, episode_map)
            fold_id = stable_fold_key(
                [feat_row["uid"], feat_row["episode_id"], str(row_idx)],
                n_splits,
            )

            keys = {
                "uid": feat_row["uid"],
                "episode_id": feat_row["episode_id"],
                "tab_name": feat_row["tab_name"],
                "scene_name": feat_row["scene_name"],
                "entrance_type": feat_row["entrance_type"],
                "uid_tab": f'{feat_row["uid"]}||{feat_row["tab_name"]}',
                "uid_entrance": f'{feat_row["uid"]}||{feat_row["entrance_type"]}',
                "episode_tab": f'{feat_row["episode_id"]}||{feat_row["tab_name"]}',
                "episode_entrance": f'{feat_row["episode_id"]}||{feat_row["entrance_type"]}',
            }

            for name, key in keys.items():
                add_freq(freq[name], key)
                add_target_stat(te_total[name], te_fold[name], key, fold_id, label)

    global_mean = label_sum / row_count if row_count else 0.5
    return freq, te_total, te_fold, global_mean


def attach_stats(row, row_idx, freq, te_total, te_fold, global_mean, n_splits, is_train):
    fold_id = stable_fold_key([row["uid"], row["episode_id"], str(row_idx)], n_splits)
    keys = {
        "uid": row["uid"],
        "episode_id": row["episode_id"],
        "tab_name": row["tab_name"],
        "scene_name": row["scene_name"],
        "entrance_type": row["entrance_type"],
        "uid_tab": f'{row["uid"]}||{row["tab_name"]}',
        "uid_entrance": f'{row["uid"]}||{row["entrance_type"]}',
        "episode_tab": f'{row["episode_id"]}||{row["tab_name"]}',
        "episode_entrance": f'{row["episode_id"]}||{row["entrance_type"]}',
    }

    row["uid_count"] = freq["uid"].get(keys["uid"], 0)
    row["episode_count"] = freq["episode_id"].get(keys["episode_id"], 0)
    row["tab_name_count"] = freq["tab_name"].get(keys["tab_name"], 0)
    row["scene_name_count"] = freq["scene_name"].get(keys["scene_name"], 0)
    row["entrance_type_count"] = freq["entrance_type"].get(keys["entrance_type"], 0)
    row["uid_tab_count"] = freq["uid_tab"].get(keys["uid_tab"], 0)
    row["uid_entrance_count"] = freq["uid_entrance"].get(keys["uid_entrance"], 0)
    row["episode_tab_count"] = freq["episode_tab"].get(keys["episode_tab"], 0)
    row["episode_entrance_count"] = freq["episode_entrance"].get(keys["episode_entrance"], 0)

    for name, out_name in (
        ("uid", "te_uid"),
        ("episode_id", "te_episode_id"),
        ("tab_name", "te_tab_name"),
        ("scene_name", "te_scene_name"),
        ("entrance_type", "te_entrance_type"),
        ("uid_tab", "te_uid_tab"),
        ("uid_entrance", "te_uid_entrance"),
        ("episode_tab", "te_episode_tab"),
        ("episode_entrance", "te_episode_entrance"),
    ):
        if is_train:
            te_value, te_count = get_oof_te(te_total[name], te_fold[name], keys[name], fold_id, global_mean)
        else:
            te_value, te_count = get_test_te(te_total[name], keys[name], global_mean)
        row[out_name] = round(te_value, 6)
        row[f"{out_name}_train_count"] = te_count

    row["global_label_mean"] = round(global_mean, 6)
    return row


def write_features(
    input_path,
    output_path,
    user_map,
    episode_map,
    freq,
    te_total,
    te_fold,
    global_mean,
    n_splits,
    is_train,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    if is_train:
        fields.extend(["row_id", "label"])
    else:
        fields.append("id")
    fields.extend(feature_fields())

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for row_idx, raw_row in enumerate(reader):
            feat_row = build_feature_row(raw_row, user_map, episode_map)
            feat_row = attach_stats(
                feat_row,
                row_idx=row_idx,
                freq=freq,
                te_total=te_total,
                te_fold=te_fold,
                global_mean=global_mean,
                n_splits=n_splits,
                is_train=is_train,
            )
            if is_train:
                feat_row["row_id"] = row_idx
                feat_row["label"] = int(raw_row["label"])
            else:
                feat_row["id"] = raw_row["id"]
            writer.writerow({field: feat_row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description="Build non-leaking baseline features for IFTech.")
    parser.add_argument("--data-dir", default="IFTech/data", help="Directory containing raw csv files.")
    parser.add_argument(
        "--output-dir",
        default="IFTech/feature",
        help="Directory for generated baseline feature csv files.",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Number of OOF folds.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    user_path = data_dir / "user_feature.csv"
    episode_path = data_dir / "episode_feature.csv"
    additional_path = data_dir / "episode_additional.csv"

    print("Collecting needed episode ids from train/test ...")
    needed_episode_ids = collect_needed_episode_ids(train_path, test_path)
    print("Needed episode ids:", len(needed_episode_ids))

    print("Loading user features ...")
    user_map = load_user_features(user_path)
    print("Loaded users:", len(user_map))

    print("Loading filtered episode-side features ...")
    episode_map = load_episode_side_features(episode_path, additional_path, needed_episode_ids)
    print("Loaded episodes:", len(episode_map))

    print("Building train-side frequency and OOF target stats ...")
    freq, te_total, te_fold, global_mean = build_stats(
        train_path=train_path,
        user_map=user_map,
        episode_map=episode_map,
        n_splits=args.n_splits,
    )
    print("Global label mean:", round(global_mean, 6))

    train_out = output_dir / "train_features_baseline.csv"
    test_out = output_dir / "test_features_baseline.csv"

    print("Writing train features ...")
    write_features(
        input_path=train_path,
        output_path=train_out,
        user_map=user_map,
        episode_map=episode_map,
        freq=freq,
        te_total=te_total,
        te_fold=te_fold,
        global_mean=global_mean,
        n_splits=args.n_splits,
        is_train=True,
    )

    print("Writing test features ...")
    write_features(
        input_path=test_path,
        output_path=test_out,
        user_map=user_map,
        episode_map=episode_map,
        freq=freq,
        te_total=te_total,
        te_fold=te_fold,
        global_mean=global_mean,
        n_splits=args.n_splits,
        is_train=False,
    )

    print("Done.")
    print("Train features:", train_out)
    print("Test features:", test_out)


if __name__ == "__main__":
    main()
