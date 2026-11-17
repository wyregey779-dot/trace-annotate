"""Check that a candidate decision is a boundary for the editor's undo controls."""

import os
from pathlib import Path

import pytest


UI = Path(__file__).resolve().parents[1] / "trace_annotate/workspace/ui.py"


def test_candidate_decision_clears_prior_undo_and_redo_without_disabling_later_edits():
    playwright = pytest.importorskip("playwright.sync_api")
    source = UI.read_text(encoding="utf-8")
    history = source[source.index("const taskUndoStacks="):source.index("async function importTemporalFile(")]
    bootstrap = """
      const state={task:{id:'task-1'},busy:false,pending:null,loadingTask:false};
      const $=id=>document.getElementById(id);
      const calls=[];
      const mayLeave=async()=>true;
      const notice=()=>{};
      async function saveCommand(operation,payload,options){
        calls.push({operation,payload,options});
        const command_id='receipt-'+calls.length;
        recordHistory({command_id},{taskId:'task-1',selectedId:null,
          command:{operation},history:options?.history||null});
        return {revision:calls.length};
      }
    """ + history + """
      window.fixture={state,calls,recordHistory,performUndo,performRedo,
        stack:()=>({undo:getUndoStack('task-1').length,redo:getRedoStack('task-1').length})};
      document.getElementById('undo-btn').addEventListener('click',performUndo);
      document.getElementById('redo-btn').addEventListener('click',performRedo);
    """
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, channel="chrome" if os.name == "nt" else None)
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content('<button id="undo-btn" disabled>撤销</button><button id="redo-btn" disabled>重做</button>')
            page.add_script_tag(content=bootstrap)
            page.evaluate("fixture.recordHistory({command_id:'add-before'},"
                          "{taskId:'task-1',selectedId:null,command:{operation:'add_annotation'}})")
            assert page.locator("#undo-btn").is_enabled()
            page.evaluate("fixture.recordHistory({command_id:'decision'},"
                          "{taskId:'task-1',selectedId:null,command:{operation:'decide_candidate'}})")
            assert page.evaluate("fixture.stack()") == {"undo": 0, "redo": 0}
            assert page.locator("#undo-btn").is_disabled()
            assert page.locator("#redo-btn").is_disabled()
            page.evaluate("fixture.performUndo()")
            assert page.evaluate("fixture.calls.length") == 0
            page.evaluate("fixture.recordHistory({command_id:'edit-after'},"
                          "{taskId:'task-1',selectedId:null,command:{operation:'update_annotation'}})")
            assert page.locator("#undo-btn").is_enabled()
            page.locator("#undo-btn").click()
            assert page.evaluate("fixture.calls[0].payload.target_command_id") == "edit-after"
            assert page.locator("#redo-btn").is_enabled()
            page.locator("#redo-btn").click()
            assert page.evaluate("fixture.calls[1].payload.target_command_id") == "receipt-1"
            assert not errors, errors
        finally:
            browser.close()
