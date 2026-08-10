# Keeping an index fresh automatically

Two ways, depending on how quickly you need changes to show up.

## A timer, re-indexing on a schedule

Indexing is incremental, so a scheduled run over an unchanged directory does almost
nothing: it stats each file and stops there. This is the cheaper option and the one to
prefer for a directory you edit occasionally.

```bash
mkdir -p ~/.config/systemd/user
cp npurag-index@.service npurag-index@.timer ~/.config/systemd/user/

# The instance name is the escaped path you want indexed.
systemctl --user enable --now "npurag-index@$(systemd-escape ~/Documents).timer"

systemctl --user list-timers 'npurag-*'
journalctl --user -u "npurag-index@$(systemd-escape ~/Documents).service"
```

The unit assumes `npurag` is on the path at `~/.cargo/bin/npurag`, which is where
`cargo install --path .` puts it. Adjust `ExecStart` if you installed it elsewhere.

Both units run the index at idle CPU and IO priority. Background work should lose every
race against whatever you are actually doing.

## `npurag watch`, re-indexing as you type

```bash
npurag watch ~/Documents
```

Changes are debounced, so one save triggers one re-index rather than one per filesystem
event. This keeps a process running and is the right choice when the index needs to be
current within seconds; the timer is the right choice otherwise.
