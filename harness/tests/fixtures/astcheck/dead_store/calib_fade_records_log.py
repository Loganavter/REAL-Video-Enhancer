# A dead store in a harness instrument: `log` is assigned the fade comparison's log path, never read, then
# overwritten by the startup log two rows later. The astcheck rule exists because this class of bug made two
# variants of a kernel measure identical values and a correct kernel get rejected.
# provenance: harness/scripts/calib_fade_records.py:34-46 @ f0c47ca5
# provenance-note: cut at the merge commit this fixture landed on (still live in the tree at that commit); the
#   rule's founding case is the `buf = quantise(pre_relu_head)` store cited in harness/astcheck.py's docstring
#   (commit 56328bf6), whose probe script is not in this clone's object store - see
#   dead_store/quantise_pre_relu_head.py for that shape, declared NONE.
#@check: dead store
#@findings: 1

log = "harness/docs/wsp/verified/logs/calib-fade/inter25s_seq_r1_ffv1.log"
pop = Population.of("frames", tuple(f"inter25s#seq#{i}" for i in range(266)), "Interstellar cut, pure sequential")
j = json.load(open(os.path.join(a.dir, "fade-seq.json")))["summary"]
st.emit(Record(value=float(j["worst_share"]), unit="%", population=pop,
               instrument="channel-max |delta| over 64px blocks (gate_sweep --reference-kind config)",
               config="pure sequential first-48-frame calibration against the spread decode, clip opens on black",
               session=st.session, statistic="worst_of", tier="measured", id=f"FD{n:03d}",
               run="harness/scripts/wall_ab.sh + harness/scripts/gate_sweep.py",
               artifact=os.path.join(a.dir, "fade-seq.json"),
               extra={"artifact_sha256": sha(os.path.join(a.dir, "fade-seq.json")),
                      "frames_above_bar": j["frames_above_bar"], "peak": j["worst_peak"]}))
n += 1
log = "harness/docs/wsp/verified/logs/startup-timed-hybrid.log"
