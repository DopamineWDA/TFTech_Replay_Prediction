import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent / "data"


def pct(a, b):
    return round(a * 100 / b, 2) if b else None


def rate(a, b):
    return round(a / b, 6) if b else None


def qtiles(values):
    values = sorted(values)
    if not values:
        return {}
    idx = lambda q: min(len(values) - 1, int(q * (len(values) - 1)))
    return {
        "min": values[0],
        "p25": values[idx(0.25)],
        "p50": values[idx(0.50)],
        "p75": values[idx(0.75)],
        "p90": values[idx(0.90)],
        "max": values[-1],
    }


def main():
    train = BASE_DIR / "train.csv"
    test = BASE_DIR / "test.csv"
    userf = BASE_DIR / "user_feature.csv"
    epf = BASE_DIR / "episode_feature.csv"
    epa = BASE_DIR / "episode_additional.csv"

    print("=== TRAIN BASIC ===")
    label = Counter()
    miss = Counter()
    tab = Counter()
    scene = Counter()
    entrance = Counter()
    uid_count = Counter()
    epi_count = Counter()
    by_tab = defaultdict(lambda: [0, 0])
    by_scene = defaultdict(lambda: [0, 0])
    by_entrance = defaultdict(lambda: [0, 0])
    rows = 0

    with train.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            y = int(row["label"])
            label[y] += 1
            uid_count[row["uid"]] += 1
            epi_count[row["episode_id"]] += 1
            for col in ("tab_name", "scene_name", "entrance_type"):
                if row[col] == "":
                    miss[col] += 1
            tab[row["tab_name"]] += 1
            scene[row["scene_name"]] += 1
            entrance[row["entrance_type"]] += 1
            by_tab[row["tab_name"]][0] += 1
            by_tab[row["tab_name"]][1] += y
            by_scene[row["scene_name"]][0] += 1
            by_scene[row["scene_name"]][1] += y
            by_entrance[row["entrance_type"]][0] += 1
            by_entrance[row["entrance_type"]][1] += y

    print("rows", rows)
    print("label_counts", dict(label))
    print("label_rate_1", rate(label[1], rows))
    print("unique_uids", len(uid_count))
    print("unique_episode_ids", len(epi_count))
    print("missing", {k: {"cnt": v, "pct": pct(v, rows)} for k, v in miss.items()})
    print("uid_freq_qtiles", qtiles(list(uid_count.values())))
    print("episode_freq_qtiles", qtiles(list(epi_count.values())))

    for name, counter, grouped in (
        ("tab_name", tab, by_tab),
        ("scene_name", scene, by_scene),
        ("entrance_type", entrance, by_entrance),
    ):
        print(f"\n=== TRAIN {name.upper()} TOP ===")
        for key, cnt in counter.most_common(12):
            n, s = grouped[key]
            print(repr(key), "cnt", cnt, "pct", pct(cnt, rows), "pos_rate", rate(s, n))

    print("\n=== TEST BASIC ===")
    miss_t = Counter()
    test_uids = set()
    test_eps = set()
    test_rows = 0
    with test.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_rows += 1
            test_uids.add(row["uid"])
            test_eps.add(row["episode_id"])
            for col in ("tab_name", "scene_name", "entrance_type"):
                if row[col] == "":
                    miss_t[col] += 1
    print("rows", test_rows)
    print("unique_uids", len(test_uids))
    print("unique_episode_ids", len(test_eps))
    print("missing", {k: {"cnt": v, "pct": pct(v, test_rows)} for k, v in miss_t.items()})
    print("uid_overlap_with_train_pct", pct(len(set(uid_count) & test_uids), len(test_uids)))
    print("episode_overlap_with_train_pct", pct(len(set(epi_count) & test_eps), len(test_eps)))

    print("\n=== USER FEATURE ===")
    miss_u = Counter()
    sex = Counter()
    addr = Counter()
    rg = Counter()
    ages = []
    spans = []
    user_ids = set()
    fmt = "%Y-%m-%d"
    with userf.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user_ids.add(row["uid"])
            sex[row["sex"]] += 1
            addr[row["address"]] += 1
            rg[row["rg_source"]] += 1
            for k, v in row.items():
                if v == "":
                    miss_u[k] += 1
            if row["age"].isdigit():
                ages.append(int(row["age"]))
            try:
                spans.append(
                    (
                        datetime.strptime(row["exp_date"], fmt)
                        - datetime.strptime(row["rg_date"], fmt)
                    ).days
                )
            except ValueError:
                pass
    print("rows", len(user_ids))
    print("coverage_train_uid_pct", pct(len(set(uid_count) & user_ids), len(uid_count)))
    print("coverage_test_uid_pct", pct(len(test_uids & user_ids), len(test_uids)))
    print("missing", dict(miss_u))
    print("sex", sex.most_common())
    print("top_address", addr.most_common(10))
    print("rg_source", rg.most_common())
    print("age_qtiles", qtiles(ages))
    print("membership_span_days_qtiles", qtiles(spans))

    print("\n=== EPISODE FEATURE ===")
    miss_ep = Counter()
    lang = Counter()
    duration = []
    cat_cnt = []
    host_cnt = []
    producer_cnt = []
    writer_cnt = []
    ep_ids = set()
    with epf.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep_ids.add(row["episode_id"])
            for k, v in row.items():
                if v == "":
                    miss_ep[k] += 1
            lang[row["language"]] += 1
            if row["duration"].isdigit():
                duration.append(int(row["duration"]))
            cat_cnt.append(0 if row["category_ids"] == "" else row["category_ids"].count("|") + 1)
            host_cnt.append(row["host"].count("|") + 1)
            producer_cnt.append(row["producer"].count("|") + 1)
            writer_cnt.append(row["writer"].count("|") + 1)
    print("rows", len(ep_ids))
    print("coverage_train_episode_pct", pct(len(set(epi_count) & ep_ids), len(epi_count)))
    print("coverage_test_episode_pct", pct(len(test_eps & ep_ids), len(test_eps)))
    print("missing", dict(miss_ep))
    print("language_top10", lang.most_common(10))
    print("duration_ms_qtiles", qtiles(duration))
    print("category_count_qtiles", qtiles(cat_cnt))
    print("host_count_qtiles", qtiles(host_cnt))
    print("producer_count_qtiles", qtiles(producer_cnt))
    print("writer_count_qtiles", qtiles(writer_cnt))

    print("\n=== EPISODE ADDITIONAL ===")
    uuid_count = Counter()
    title_tok = []
    title_len = []
    add_ids = set()
    with epa.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            add_ids.add(row["episode_id"])
            uuid_count[row["uuid"]] += 1
            title_tok.append(row["title"].count("|") + 1)
            title_len.append(len(row["title"]))
    print("rows", len(add_ids))
    print("coverage_train_episode_pct", pct(len(set(epi_count) & add_ids), len(epi_count)))
    print("coverage_test_episode_pct", pct(len(test_eps & add_ids), len(test_eps)))
    print("uuid_unique", len(uuid_count))
    print("shared_uuid_pct", pct(sum(v > 1 for v in uuid_count.values()), len(uuid_count)))
    print("title_token_count_qtiles", qtiles(title_tok))
    print("title_char_len_qtiles", qtiles(title_len))


if __name__ == "__main__":
    main()
