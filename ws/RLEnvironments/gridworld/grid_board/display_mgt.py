import os
import tkinter

import time
from collections import namedtuple

from PIL import ImageTk, Image

from ws.RLEnvironments.gridworld import CONFIG
from ws.RLUtils.common.misc_functions import calc_pixels

PhotoImage = ImageTk.PhotoImage

COORD_LEFT = (7, 42)  # left
COORD_RIGHT = (77, 42)  # right
COORD_UP = (42, 5)  # up
COORD_DOWN = (42, 77)  # down

LOW_NUMBER = -999999

def display_mgt(strategy, env):

    _tk = tkinter.Tk()
    config = CONFIG.fn_get_config()

    _display_info = config.DISPLAY
    _unit = _display_info["UNIT"]
    _width = _display_info['WIDTH']
    _height = _display_info['HEIGHT']
    _canvas = tkinter.Canvas(
        height=_height * _unit + 25,
        width=_width * _unit,
        # bg= 'white'
    )

    _cursor = None

    _board_blockers = _display_info['BOARD_BLOCKERS']
    _board_goal = _display_info['BOARD_GOAL']
    _right_margin = 5
    _bottom_margin = 80

    _test_mode = False

    _actions = {}
    def fn_close():
        _tk.destroy()


    def fn_set_test_mode():
        nonlocal _test_mode
        _test_mode = True

    def fn_create_value_repo():
        return [[0.0] * _width for _ in range(_width)]

    def fn_compare_value_repos(repo1, repo2):
        for col in range(0, _height):
            for row in range(0, _width):
                if repo1[col][row] != repo2[col][row]:
                    repo1[col][row] = repo2[col][row]
                    return True
        return False

    def fn_get_state_actions(state, action_size, fn_get_state_value):

        row, col = state
        action_size = action_size
        possible_actions = [LOW_NUMBER] * action_size

        dir_up = [row, max(0, col -1)]
        if col > 0:
            possible_actions[0] = fn_get_state_value(dir_up)

        dir_down = [row, min(_height - 1, col + 1) ]
        if col < _height - 1:
            possible_actions[1] = fn_get_state_value(dir_down)

        dir_left = [max(0, row - 1), col]
        if row > 0:
            possible_actions[2] = fn_get_state_value(dir_left)

        dir_right = [ min(_width - 1, row + 1) , col]
        if row < _width - 1:
            possible_actions[3] = fn_get_state_value(dir_right)

        return possible_actions

    def _fn_get_app_title():
        strategy_parts = strategy.rsplit('.')
        app_type = strategy_parts[len(strategy_parts) - 1]
        app_title = app_type.replace('_', ' ')
        return app_title

    def canvas_text_mgt(canvas):
        _dict = {}

        def fn_push(key, val):
            nonlocal _dict

            if key in _dict:
                lst_of_refs = _dict.pop(key)
                for ref in lst_of_refs:
                    canvas.delete(ref)
            _dict[key] = val

        return fn_push

    _fn_filter_canvas_text = canvas_text_mgt(_canvas)

    def _fn_calculate_step(state, newState):
        stepX, stepY = newState[0] - state[0], newState[1] - state[1]
        return stepX, stepY

    def _fn_show_qvalue_directions(state, stateAction, coord):
        x = coord[0] + _unit * state[0]
        y = coord[1] + _unit * state[1]
        val = round(stateAction, 2)
        font = ('Helvetica', str(10), 'normal')
        text_ref = _tk.canvas.create_text(x, y, fill="black", text=str(val),
                                               font=font, anchor="nw")
        return text_ref

    def _fn_append_rewards_to_canvas():
        _fn_append_reward_to_canvas(_board_goal['x'], _board_goal['y'], str(_board_goal['reward']))
        for blocker in _board_blockers:
            _fn_append_reward_to_canvas(blocker['x'], blocker['y'], str(blocker['reward']))

    def _fn_append_reward_to_canvas(row, col, contents, font='Helvetica', size=10,
                             style='bold', anchor="nw"):
        origin_x, origin_y = 45, 50
        x, y = origin_x + (_unit * row), origin_y + (_unit * col)
        font = (font, str(size), style)
        text = _tk.canvas.create_text(x, y, fill="yellow", text=contents,
                                           font=font, anchor=anchor)
        _tk.texts.append(text)

    def _fn_render_on_canvas():
        time.sleep(0.1)
        _tk.canvas.tag_raise(_cursor)
        _tk.update()

    def _fn_create_button(canvas, button_x_offset, button_name, button_action):
        bound_button = tkinter.Button(bg="white",
                                 text=button_name,
                                 command=button_action)
        bound_button.configure(width=10, height=2)
        canvas.create_window(_width * _unit * button_x_offset, _height * _unit + 45,
                             window=bound_button)


    def _fn_load_images():
        rwd = os.path.dirname(__file__)
        image_dir = os.path.join(rwd, 'img')

        up = PhotoImage(Image.open(image_dir + "/up.png").resize((13, 13)))
        right = PhotoImage(Image.open(image_dir + "/right.png").resize((13, 13)))
        left = PhotoImage(Image.open(image_dir + "/left.png").resize((13, 13)))
        down = PhotoImage(Image.open(image_dir + "/down.png").resize((13, 13)))
        rectangle = PhotoImage(Image.open(image_dir + "/penalty_box.png").resize((65, 65)))
        triangle = PhotoImage(Image.open(image_dir + "/reward_box.png").resize((65, 65)))
        circle = PhotoImage(Image.open(image_dir + "/cursor.png").resize((32, 32)))
        return (up, down, left, right), (rectangle, triangle, circle)

    def _fn_append_text_canvas(col, row, contents, font='Helvetica', size=10,
                           style='normal', anchor="nw"):
        origin_x, origin_y = 10, 85
        x, y = origin_x + (_unit * row), origin_y + (_unit * col)
        font = (font, str(size), style)
        text = _tk.canvas.create_text(x, y, fill="black", text=contents,
                                           font=font, anchor=anchor)
        _tk.texts.append(text)

    def _fn_draw_arrow(col, row, policy):

        if col == _board_goal['y'] and row == _board_goal['x']:
            return

        if policy[0] > 0:  # up
            origin_x, origin_y = 50 + (_unit * row), 10 + (_unit * col)
            _tk.arrows.append(_tk.canvas.create_image(origin_x, origin_y,
                                                                image=_tk.up))
        if policy[1] > 0:  # down
            origin_x, origin_y = 50 + (_unit * row), 90 + (_unit * col)
            _tk.arrows.append(_tk.canvas.create_image(origin_x, origin_y,
                                                                image=_tk.down))
        if policy[2] > 0:  # left
            origin_x, origin_y = 10 + (_unit * row), 50 + (_unit * col)
            _tk.arrows.append(_tk.canvas.create_image(origin_x, origin_y,
                                                                image=_tk.left))
        if policy[3] > 0:  # right
            origin_x, origin_y = 90 + (_unit * row), 50 + (_unit * col)
            _tk.arrows.append(_tk.canvas.create_image(origin_x, origin_y,
                                                                image=_tk.right))


    def fn_move_cursor(stateStart, stateEnd=(0, 0)):
        step = _fn_calculate_step(stateStart, stateEnd)

        _tk.canvas.move(_cursor, step[0] * _unit, step[1] * _unit)
        _fn_render_on_canvas()

    def fn_show_policy_arrows(policy_table, show= True):
        for i in _tk.arrows:
            _tk.canvas.delete(i)
        if show:
            for i in range(_height):
                for j in range(_width):
                    _fn_draw_arrow(i, j, policy_table[i][j])

    def fn_show_state_values(value_table, show= True):
        for i in _tk.texts:
            _tk.canvas.delete(i)
        if show:
            for i in range(_height):
                for j in range(_width):
                    val = round(value_table[i][j], 8)
                    _fn_append_text_canvas(i, j, val)
        _fn_append_rewards_to_canvas()
        _fn_render_on_canvas()

    def fn_show_qvalue(state, q_actions):
        stateStr = str(state)

        q_action_list = list(q_actions)
        refs = [_fn_show_qvalue_directions(state, q_action_list[0], COORD_UP),
                _fn_show_qvalue_directions(state, q_action_list[1], COORD_DOWN),
                _fn_show_qvalue_directions(state, q_action_list[2], COORD_LEFT),
                _fn_show_qvalue_directions(state, q_action_list[3], COORD_RIGHT)]

        _fn_filter_canvas_text(stateStr, refs)
        _fn_append_rewards_to_canvas()
        _fn_render_on_canvas()

    def fn_is_target_state_reached(state):
        if state == (_board_goal['x'], _board_goal['y']):
            return True
        return False

    def fn_get_start_state():
        return [0, 0]

    def fn_run_next_move(env, state, fnNextGetAction):
        action = fnNextGetAction(state)
        if action < 0:
            return None
        next_state = env.fn_get_allowed_moves()[action]
        new_x = state[0] + next_state[0]
        new_y = state[1] + next_state[1]
        return new_x, new_y

    def fn_update_qvalue(state, actions):
        if fn_show_qvalue is not None:
            fn_show_qvalue(state, actions)

    def fn_is_goal_reached(state):
        return True if state == [_board_goal['x'], _board_goal['y']] else False

    def _fn_build_canvas(acton_dictionary):
        nonlocal _cursor

        # create lines
        for col in range(0, (_width + 1) * _unit, _unit):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, _height * _unit
            _canvas.create_line(x0, y0, x1, y1)

        for row in range(0, (_height + 1) * _unit, _unit):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, _width * _unit, row
            _canvas.create_line(x0, y0, x1, y1)

        _cursor = _canvas.create_image(_unit / 2, _unit / 2, image=_tk.shapes[2])
        for blocker in _board_blockers:
            pix_x, pix_y = calc_pixels(_unit, blocker['x'], blocker['y'])
            _canvas.create_image(pix_x, pix_y, image=_tk.shapes[0])

        pix_x, pix_y = calc_pixels(_unit, _board_goal['x'], _board_goal['y'])
        _canvas.create_image(pix_x, pix_y, image=_tk.shapes[1])

        button_x_offset = .10
        for label, fn in acton_dictionary.items():
            _fn_create_button(_canvas, button_x_offset, label, fn)
            button_x_offset += .20

        _canvas.pack()
        return _canvas


    def fn_setup_ui(actions):
        nonlocal _actions

        _actions = actions

        _tk.geometry('{0}x{1}'.format(_width * _unit + _right_margin,
                                      _height * _unit + _bottom_margin))
        _tk.texts = []
        _tk.arrows = []

        (_tk.up, _tk.down, _tk.left, _tk.right), _tk.shapes = _fn_load_images()
        _tk.canvas = _fn_build_canvas(_actions)
        _fn_append_rewards_to_canvas()
        _fn_render_on_canvas()

    def fn_run_ui():
        if _test_mode:
            for key, action in _actions.items():
                if action is not None:
                    action()
            return
        _tk.mainloop()

    app_title = _fn_get_app_title()
    _tk.title(app_title)

    ret_obj = namedtuple('_', [
        # 'DisplayStructure',
        'Config',
        'fn_setup_ui',
        'fn_run_ui',
        'fn_move_cursor',
        'fn_show_policy_arrows',
        'fn_show_state_values',

        'fn_show_qvalue',
        'fn_is_target_state_reached',
        'fn_get_start_state',
        'fn_run_next_move',
        'fn_update_qvalue',
        'fn_is_goal_reached',
        'fn_create_value_repo',
        'fn_compare_value_repos',
        'fn_get_state_actions',
        'fn_set_test_mode',
        'fn_close',
    ])

    # ret_obj.DisplayStructure = _display_info
    ret_obj.Config = config
    ret_obj.fn_setup_ui = fn_setup_ui
    ret_obj.fn_run_ui = fn_run_ui
    ret_obj.fn_move_cursor = fn_move_cursor
    ret_obj.fn_show_policy_arrows = fn_show_policy_arrows
    ret_obj.fn_show_state_values = fn_show_state_values

    ret_obj.fn_show_qvalue = fn_show_qvalue
    ret_obj.fn_is_target_state_reached = fn_is_target_state_reached
    ret_obj.fn_get_start_state = fn_get_start_state
    ret_obj.fn_run_next_move = fn_run_next_move
    ret_obj.fn_update_qvalue = fn_update_qvalue
    ret_obj.fn_is_goal_reached = fn_is_goal_reached
    ret_obj.fn_create_value_repo = fn_create_value_repo
    ret_obj.fn_compare_value_repos = fn_compare_value_repos
    ret_obj.fn_get_state_actions = fn_get_state_actions
    ret_obj.fn_set_test_mode = fn_set_test_mode
    ret_obj.fn_close = fn_close
    return ret_obj


