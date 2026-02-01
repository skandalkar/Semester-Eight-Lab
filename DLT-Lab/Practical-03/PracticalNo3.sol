// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.7.4;

contract Persons {
    uint[] _ids;
    mapping(uint => string) _names;

    mapping(uint => bool) _exists;

    function setPerson(uint id, string memory name) public {
        if (!_exists[id]) {
            _ids.push(id);
            _exists[id] = true;
        }
        _names[id] = name;
    }

    function getKeys() public view returns (uint[] memory) {
        return _ids;
    }
}